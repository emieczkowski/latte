import asyncio
import json, re, time, subprocess
from pathlib import Path
from orchestrator.token_parser import (
    parse_actions, EditFileAction, ClaimTaskAction, CompleteTaskAction,
    RunTestsAction, RunScriptAction, ReadFileAction, AssignTaskAction, BroadcastAction,
    RequestStatusAction, SynthesizeAction, DiscoverTaskAction, ReleaseTaskAction, VerifyTaskAction,
    CloseTaskAction
)
from orchestrator.locks import LockManager
import networkx as nx



class Orchestrator:
    def __init__(self, run_dir, repo_dir, agents, task_list_path, straggler_rounds=None, dynamic_agents=False, static_graph=False, allow_run_tests=True):
        self.run_dir = run_dir
        self.repo_dir = repo_dir
        self.agents = agents
        self.task_list_path = task_list_path
        self.lock_mgr = LockManager(run_dir / ".locks")
        self.events_path = run_dir / "events.jsonl"
        self._current_round = None
        self._last_test_returncode: int | None = None
        self._allowed_graph_ops: set = set()
        self._graph: nx.DiGraph = nx.DiGraph()
        self._project_meta: dict = {}
        # Straggler mitigation
        self.straggler_rounds: int | None = straggler_rounds
        self._task_assigned_round: dict[str, int] = {} # round number when it was last assigned/claimed
        # Verification: when a task completes with exposure >= threshold, notify lead
        self.verify_exposure_threshold: float | None = None
        self._verified_tasks: set[str] = set()
        # Static graph mode: after planning, discover_task is locked out
        self.static_graph: bool = static_graph
        # Dynamic agent scaling: only dispatch agents that have work to do each round.
        # False = fixed-N (ablation baseline), True = frontier-capped dispatch.
        self.dynamic_agents: bool = dynamic_agents
        # When False, <run_tests /> is blocked — agents must write their own verification scripts.
        self.allow_run_tests: bool = allow_run_tests
        self._lead_agent_name: str | None = None
        self._agent_last_shown_task: dict[str, str | None] = {}
        self._agent_last_dispatched: dict[str, int] = {}

        # Heartbeat monitoring.
        self._agent_last_edit_round: dict[str, int] = {}
        self._agent_last_script_round: dict[str, int] = {}
        self._heartbeat_alerted_tasks: set[str] = set()
        self._HEARTBEAT_STUCK_ROUNDS: int = 4
        self._round_file_writers: dict[str, str] = {}

    def load_tasks(self):
        """
        Return current task state. DiGraph is ground truth: 
        serialize from it on every call after init.
        """
        if not self._graph.nodes:
            raw = json.loads(self.task_list_path.read_text())
            self._project_meta = {k: v for k, v in raw.items() if k != "tasks"}
            self._sync_graph_from(raw["tasks"])
        return self._tasks_from_graph()

    def save_tasks(self, tasks):
        """
        Merge task-dict changes into the DiGraph (ground truth), 
        then checkpoint to disk. 
        """
        for t in tasks["tasks"]:
            if self._graph.has_node(t["id"]):
                self._graph.nodes[t["id"]].update(t)
        self.task_list_path.write_text(json.dumps(tasks, indent=2))

    def _tasks_from_graph(self) -> dict:
        """Serialize DiGraph back to the tasks dict format."""
        tasks = [dict(self._graph.nodes[n]) for n in self._graph.nodes]
        return {**self._project_meta, "tasks": tasks}

    def _sync_graph_from(self, task_list: list):
        """Populate DiGraph from a task list. Called once at init."""
        self._graph.clear()
        for t in task_list:
            self._graph.add_node(t["id"], **t)
        for t in task_list:
            for dep in t.get("dependencies", []):
                if self._graph.has_node(dep):
                    self._graph.add_edge(dep, t["id"])

    def log(self, event):
        event["ts"] = time.time()
        if self._current_round is not None:
            event.setdefault("round", self._current_round)
        with self.events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    _STDLIB_SHADOWS = frozenset({
        "ast", "collections", "copy", "csv", "datetime", "decimal", "difflib",
        "enum", "fnmatch", "fractions", "functools", "gc", "glob", "hashlib",
        "heapq", "hmac", "html", "http", "importlib", "inspect", "io",
        "itertools", "json", "keyword", "linecache", "locale", "logging",
        "math", "mimetypes", "numbers", "operator", "os", "pathlib", "pickle",
        "platform", "pprint", "queue", "random", "re", "shutil", "signal",
        "socket", "sqlite3", "stat", "string", "struct", "subprocess",
        "sys", "tempfile", "textwrap", "threading", "time", "traceback",
        "types", "typing", "unicodedata", "unittest", "urllib", "uuid",
        "warnings", "weakref", "xml", "zipfile",
    })

    def apply_edit(self, agent_name, action):
        rel = action.path
        # Weird agent prob where sometimes they preface with "repo/" even though repo_dir is already root
        if rel.startswith("repo/"):
            rel = rel[len("repo/"):]

        # Block writes that would shadow stdlib modules (e.g. inspect.py, re.py)
        stem = Path(rel).stem
        if stem in self._STDLIB_SHADOWS:
            self.log({"type": "edit_blocked", "agent": agent_name, "path": rel,
                      "reason": "stdlib_shadow"})
            return False, "STDLIB_SHADOW"

        abs_path = self.repo_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        prior_writer = self._round_file_writers.get(rel)
        if prior_writer and prior_writer != agent_name:
            self.log({"type": "edit_blocked", "agent": agent_name, "path": rel,
                      "reason": "file_locked", "locked_by": prior_writer})
            return False, f"FILE_LOCKED_BY_{prior_writer}"
        self._round_file_writers[rel] = agent_name

        content = action.content

        content = re.sub(r'^```[^\n]*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        abs_path.write_text(content)

        self._agent_last_edit_round[agent_name] = self._current_round or 0
        self.log({"type": "edit_file", "agent": agent_name, "path": rel, "n_chars": len(action.content)})
        return True, "OK"

    def run_tests(self):
        t0 = time.time()
        proc = subprocess.run(["pytest", "-q"], cwd=self.repo_dir, capture_output=True, text=True)
        dt = time.time() - t0
        self.log({
            "type": "run_tests",
            "returncode": proc.returncode,
            "seconds": dt,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        })
        self._last_test_stdout = proc.stdout
        self._last_test_returncode = proc.returncode
        return proc.returncode == 0

    def run_script(self, path: str):
        t0 = time.time()
        proc = subprocess.run(["python", path], cwd=self.repo_dir, capture_output=True, text=True)
        dt = time.time() - t0
        self.log({
            "type": "run_script",
            "path": path,
            "returncode": proc.returncode,
            "seconds": dt,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        })
        output = proc.stdout[-3000:] + (f"\n[stderr]\n{proc.stderr[-1000:]}" if proc.stderr.strip() else "")
        return proc.returncode, output

    def get_task_summary(self, tasks):
        """Full task summary for the lead agent. Done tasks are collapsed to a count."""
        summary = ["=== CURRENT TASKS ===\n"]
        done_count = sum(1 for t in tasks["tasks"] if t["status"] == "done")
        if done_count:
            summary.append(f"[{done_count} completed task(s) — not shown]\n")
        for t in tasks["tasks"]:
            if t["status"] == "done":
                continue
            status_icon = {"pending": "⏳", "assigned": "📌", "in_progress": "🔨"}.get(t["status"], "❓")
            owner_info = f" (Owner: {t['owner']})" if t.get("owner") else " (Unassigned)"
            deps = f" [Requires: {', '.join(t['dependencies'])}]" if t.get("dependencies") else ""
            summary.append(f"{status_icon} {t['id']}: {t['title']}{owner_info}{deps}")

        ready = self.get_ready_tasks()
        if ready:
            summary.append("\n=== READY TO ASSIGN (deps satisfied) ===")
            for task_id in ready:
                node = self._graph.nodes[task_id]
                summary.append(f"  • {task_id}: {node['title']}")

        path = self.critical_path()
        if path:
            summary.append(f"\nCritical path: {' → '.join(path)}")

        return "\n".join(summary)

    def _get_agent_context(self, agent_name: str, tasks: dict, is_lead: bool = False) -> str:
        """
        Lead sees the full task graph.
        Teammates see only their current task, or the list of ready tasks if idle.
        This keeps teammate context windows small regardless of graph size.
        """
        if is_lead:
            return self.get_task_summary(tasks)

        agent = self.agents[agent_name]
        current_task_id = agent.current_task

        if current_task_id:
            task = next((t for t in tasks["tasks"] if t["id"] == current_task_id), None)
            if task:
                return (
                    f"=== YOUR CURRENT TASK ===\n"
                    f"ID: {task['id']}  Status: {task['status']}\n"
                    f"Title: {task['title']}\n\n"
                    f"{task['description']}"
                )

        ready = self.get_ready_tasks()
        if not ready:
            return "No tasks are currently available. Wait for dependencies to complete."

        lines = ["=== AVAILABLE TASKS (ready to claim) ==="]
        for task_id in ready:
            node = self._graph.nodes[task_id]
            desc_preview = node.get("description", "")[:600]
            lines.append(f"\n• {task_id}: {node['title']}\n  {desc_preview}")
        return "\n".join(lines)

    def get_status_report(self):
        report = ["=== AGENT STATUS ===\n"]
        for name, agent in self.agents.items():
            status = agent.get_status()
            task_info = f"Working on: {status['current_task']}" if status['current_task'] else "Idle"
            report.append(f"• {name}: {task_info}")
        return "\n".join(report)

    def broadcast_message(self, sender, message):
        for name, agent in self.agents.items():
            if name != sender:
                agent.receive(message, sender=sender)
        self.log({"type": "broadcast", "sender": sender, "message": message[:500]})

    def assign_task(self, assigner, task_id, assignee, tasks=None):
        """Assign a task to a specific agent. Mutates tasks in place; caller is responsible for saving."""
        if tasks is None:
            tasks = self.load_tasks()
            save_after = True
        else:
            save_after = False
        for t in tasks["tasks"]:
            if t["id"] == task_id and t["status"] == "pending":
                t["owner"] = assignee
                t["assigned_by"] = assigner
                t["status"] = "assigned"
                self._graph.nodes[task_id].update({"owner": assignee, "assigned_by": assigner, "status": "assigned"})

                if assignee in self.agents:
                    msg = f"You have been assigned task '{task_id}': {t['title']}\n{t['description']}"
                    self.agents[assignee].receive(msg, sender=assigner)
                    self.agents[assignee].claim_task(task_id)

                self._task_assigned_round[task_id] = self._current_round or 0
                self.log({"type": "assign_task", "task": task_id, "assigner": assigner, "assignee": assignee})
                break
        if save_after:
            self.save_tasks(tasks)

    def release_task(self, task_id, reason="manual", requester="Lead", tasks=None):
        """Reset an in-progress or assigned task back to pending, freeing its owner."""
        save_after = tasks is None
        if tasks is None:
            tasks = self.load_tasks()
        for t in tasks["tasks"]:
            if t["id"] == task_id and t["status"] in ("assigned", "in_progress"):
                prev_owner = t.get("owner")
                t["status"] = "pending"
                t["owner"] = None
                t["assigned_by"] = None
                self._graph.nodes[task_id].update(t)
                if prev_owner and prev_owner in self.agents:
                    self.agents[prev_owner].complete_task()  
                self._task_assigned_round.pop(task_id, None)
                self.log({
                    "type": "release_task",
                    "task": task_id,
                    "prev_owner": prev_owner,
                    "requester": requester,
                    "reason": reason,
                })
                if save_after:
                    self.save_tasks(tasks)
                return True
        return False

    def _check_stragglers(self) -> bool:
        """Auto-release tasks that have been assigned/in-progress for too long."""
        if self.static_graph or self.straggler_rounds is None or self._current_round is None:
            return False
        tasks = self.load_tasks()
        released_any = False
        for t in tasks["tasks"]:
            if t["status"] not in ("assigned", "in_progress"):
                continue
            assigned_at = self._task_assigned_round.get(t["id"])
            if assigned_at is None:
                continue
            if (self._current_round - assigned_at) >= self.straggler_rounds:
                self.release_task(t["id"], reason="straggler_timeout", requester="Orchestrator")
                released_any = True
                # Notify the lead
                if "Lead" in self.agents:
                    self.agents["Lead"].receive(
                        f"Task {t['id']} ({t['title']}) was held by {t.get('owner')} "
                        f"for {self.straggler_rounds} rounds without completing and has been released. "
                        f"Please reassign it.",
                        sender="System",
                    )
        return released_any

    def exposure_score(self, task_id: str) -> float:
        """
        Fraction of remaining incomplete nodes that are downstream of task_id.
        """
        incomplete = {
            n for n in self._graph.nodes
            if self._graph.nodes[n].get("status") != "done"
            and n != task_id
        }
        if not incomplete:
            return 0.0
        descendants = nx.descendants(self._graph, task_id)
        return len(descendants & incomplete) / len(incomplete)

    def create_verify_node(self, task_id: str, requester: str = "Lead") -> str | None:
        """
        Insert a verification node for task_id into the live graph.
        The verify node depends on task_id being done, and all nodes that
        previously depended on task_id now also depend on the verify node.
        """
        verify_id = f"{task_id}-verify"
        if self._graph.has_node(verify_id) or task_id in self._verified_tasks:
            return None
        if not self._graph.has_node(task_id):
            self.log({"type": "verify_error", "task_id": task_id,
                      "msg": "verify requested for unknown task_id"})
            return None

        original = self._graph.nodes[task_id]
        successors = list(self._graph.successors(task_id))

        verify_node = {
            "id": verify_id,
            "title": f"Verify: {original.get('title', task_id)}",
            "description": (
                f"Review and verify the output of task {task_id} ({original.get('title', '')}).\n\n"
                f"Check that: (1) the implementation is correct and complete, "
                f"(2) edge cases are handled, (3) outputs match what downstream tasks expect.\n\n"
                f"If you find problems, fix them or broadcast to the team. "
                f"When satisfied, emit <complete_task id=\"{verify_id}\" />."
            ),
            "status": "pending",
            "owner": None,
            "assigned_by": None,
            "priority": original.get("priority", 1),
            "dependencies": [task_id],
            "is_verify": True,
            "files": [],
        }

        self._graph.add_node(verify_id, **verify_node)
        self._graph.add_edge(task_id, verify_id)

        for s in successors:
            self._graph.add_edge(verify_id, s)

        self._verified_tasks.add(task_id)
        tasks = self._tasks_from_graph()
        self.task_list_path.write_text(json.dumps(tasks, indent=2))
        self.log({
            "type": "create_verify_node",
            "task": task_id,
            "verify_id": verify_id,
            "requester": requester,
            "successors_rerouted": successors,
        })
        return verify_id

    def _notify_if_high_exposure(self, task_id: str):
        """After a task completes, alert the lead if its exposure warrants verification (optional param)."""
        if self.verify_exposure_threshold is None:
            return
        if task_id in self._verified_tasks:
            return
        score = self.exposure_score(task_id)
        self.log({"type": "exposure_score", "task": task_id, "score": round(score, 3)})
        if score >= self.verify_exposure_threshold and "Lead" in self.agents:
            self.log({
                "type": "verify_suggested",
                "task": task_id,
                "score": round(score, 3),
                "threshold": self.verify_exposure_threshold,
            })
            self.agents["Lead"].receive(
                f"Task {task_id} just completed with exposure score {score:.2f} "
                f"(threshold {self.verify_exposure_threshold}). Consider verifying it with "
                f"<verify_task id=\"{task_id}\" /> before downstream tasks proceed.",
                sender="System",
            )

    def _deps_satisfied(self, task_id):
        return all(
            self._graph.nodes[dep].get("status") == "done"
            for dep in self._graph.predecessors(task_id)
        )

    # def _deps_satisfied(self, task, all_tasks):
    #     """Return True if all dependencies of task are done."""
    #     task_status = {t["id"]: t["status"] for t in all_tasks}
    #     return all(task_status.get(dep) == "done" for dep in task.get("dependencies", []))

    def _process_actions(self, agent_name, actions, tasks):
        """Execute parsed actions, updating tasks in place. Returns updated tasks."""
        agent = self.agents[agent_name]
        is_lead = agent_name == self._lead_agent_name

        # Deduplicate repeated identical actions (e.g. GPT spamming <complete_task>).
        # Keep first occurrence of each (type, key-fields) tuple so order is preserved.
        seen = set()
        deduped = []
        for a in actions:
            key = (type(a).__name__,) + tuple(
                tuple(v) if isinstance(v, list) else v
                for v in vars(a).values()
            )
            if key not in seen:
                seen.add(key)
                deduped.append(a)
        actions = deduped

        for a in actions:
            if isinstance(a, (ClaimTaskAction, CompleteTaskAction, EditFileAction)) and is_lead:
                self.log({"type": "lead_task_action_blocked",
                          "agent": agent_name, "action": type(a).__name__})
                continue

            if isinstance(a, ClaimTaskAction):
                if self.static_graph:
                    self.log({"type": "claim_blocked", "agent": agent_name, "task": a.task_id,
                              "reason": "static_graph"})
                    agent.receive(
                        "⚠️ Self-assignment is not available in this run. Wait for the lead to assign tasks.",
                        sender="System",
                    )
                    continue
                for t in tasks["tasks"]:
                    if t["id"] == a.task_id and (
                        (t.get("owner") is None and t["status"] == "pending") or
                        (t.get("owner") == agent_name and t["status"] == "assigned")
                    ):
                        if not self._deps_satisfied(a.task_id):
                            self.log({"type": "claim_blocked", "agent": agent_name, "task": a.task_id,
                                      "reason": "dependencies_not_done"})
                            break
                        t["owner"] = agent_name
                        t["status"] = "in_progress"

                        agent.claim_task(a.task_id)
                        self._task_assigned_round[a.task_id] = self._current_round or 0
                        self.log({"type": "claim_task", "agent": agent_name, "task": a.task_id})
                        break

            elif isinstance(a, CompleteTaskAction):
                for t in tasks["tasks"]:
                    if t["id"] == a.task_id and t.get("owner") == agent_name:
                        t["status"] = "done"
                        t["owner"] = None
                        self._graph.nodes[a.task_id]["status"] = "done"
                        agent.complete_task()
                        self.log({"type": "complete_task", "agent": agent_name, "task": a.task_id})
                        self._notify_if_high_exposure(a.task_id)
                        break

            elif isinstance(a, EditFileAction):
                if not is_lead and not agent.current_task:
                    rel = a.path[len("repo/"):] if a.path.startswith("repo/") else a.path
                    abs_path = self.repo_dir / rel
                    if abs_path.exists():
                        self.log({"type": "edit_blocked", "agent": agent_name, "path": a.path,
                                  "reason": "no_active_task"})
                        agent.receive(
                            f"⚠️ Edit to existing file '{a.path}' blocked: claim a task before modifying project files.",
                            sender="System",
                        )
                    else:
                        ok, reason = self.apply_edit(agent_name, a)
                        if not ok:
                            agent.receive(f"⚠️ Edit to '{a.path}' blocked ({reason}): another agent already wrote this file this round. Retry next round.", sender="System")
                else:
                    ok, reason = self.apply_edit(agent_name, a)
                    if not ok:
                        agent.receive(f"⚠️ Edit to '{a.path}' blocked ({reason}): another agent already wrote this file this round. Retry next round.", sender="System")

            elif isinstance(a, RunTestsAction):
                if not self.allow_run_tests:
                    agent.receive(
                        "⚠️ <run_tests /> is not available in this experiment. "
                        "Write your own verification scripts with <edit_file> and run them with <run_script>.",
                        sender="System",
                    )
                else:
                    test_passed = self.run_tests()
                    result_msg = "✅ Tests passed!" if test_passed else f"❌ Tests failed:\n{self._last_test_stdout[-1500:]}"
                    agent.receive(result_msg, sender="System")

            elif isinstance(a, RunScriptAction):
                returncode, output = self.run_script(a.path)
                status = "✅ Script completed" if returncode == 0 else f"❌ Script exited with code {returncode}"
                agent.receive(f"{status}:\n{output}", sender="System")
                self._agent_last_script_round[agent_name] = self._current_round or 0

            elif isinstance(a, ReadFileAction):
                rel = a.path[len("repo/"):] if a.path.startswith("repo/") else a.path
                abs_path = self.repo_dir / rel
                if abs_path.exists():
                    content = abs_path.read_text()[:4000]
                    agent.receive(f"=== {rel} ===\n{content}", sender="System")
                else:
                    agent.receive(f"[file not found: {rel}]", sender="System")
                self.log({"type": "read_file", "agent": agent_name, "path": rel})

            elif isinstance(a, AssignTaskAction):
                self.assign_task(agent_name, a.task_id, a.assignee, tasks=tasks)

            elif isinstance(a, BroadcastAction):
                self.broadcast_message(agent_name, a.message)

            elif isinstance(a, RequestStatusAction):
                status_report = self.get_status_report()
                task_summary  = self.get_task_summary(self.load_tasks())
                agent.receive(f"{status_report}\n\n{task_summary}", sender="System")
                self.log({"type": "request_status_intercepted", "agent": agent_name})

            elif isinstance(a, SynthesizeAction):
                self.log({"type": "synthesize", "agent": agent_name, "summary": a.summary})

            elif isinstance(a, ReleaseTaskAction):
                if self.static_graph:
                    self.log({"type": "release_task_blocked", "agent": agent_name,
                              "task": a.task_id, "reason": "static_graph"})
                else:
                    self.release_task(a.task_id, reason="lead_requested", requester=agent_name, tasks=tasks)

            elif isinstance(a, CloseTaskAction):
                if not is_lead:
                    self.log({"type": "close_task_blocked", "agent": agent_name,
                              "task": a.task_id, "reason": "lead_only"})
                else:
                    for t in tasks["tasks"]:
                        if t["id"] == a.task_id and t["status"] != "done":
                            prev_owner = t.get("owner")
                            t["status"] = "done"
                            t["owner"] = None
                            self._graph.nodes[a.task_id]["status"] = "done"
                            self._graph.nodes[a.task_id]["owner"] = None
                            if prev_owner and prev_owner in self.agents:
                                self.agents[prev_owner].complete_task()
                            self._task_assigned_round.pop(a.task_id, None)
                            self.log({"type": "close_task", "agent": agent_name,
                                      "task": a.task_id, "prev_owner": prev_owner})
                            break

            elif isinstance(a, VerifyTaskAction):
                if self.static_graph:
                    self.log({"type": "verify_task_blocked", "agent": agent_name,
                              "task": a.task_id, "reason": "static_graph"})
                else:
                    verify_id = self.create_verify_node(a.task_id, requester=agent_name)
                    if verify_id and "Lead" in self.agents:
                        self.agents["Lead"].receive(
                            f"Verify node {verify_id} created for task {a.task_id}. "
                            f"Assign it to a teammate with <assign_task id=\"{verify_id}\" to=\"DevX\" />.",
                            sender="System",
                        )

            elif isinstance(a, DiscoverTaskAction):
                if type(a) in self._allowed_graph_ops:
                    self._apply_discover_task(a, tasks, agent_name)

        return tasks

    def _apply_discover_task(self, action: DiscoverTaskAction, tasks: dict, agent_name: str = "unknown"):
        """Append a new task node proposed by the lead during the planning phase."""
        existing_ids = {t["id"] for t in tasks["tasks"]}
        if action.id in existing_ids:
            self.log({"type": "discover_task_skipped", "reason": "duplicate_id", "task": action.id})
            return

        # Validate that all decl deps already exist in the graph.
        valid_deps = []
        for dep in action.dependencies:
            if dep in existing_ids:
                valid_deps.append(dep)
            else:
                self.log({"type": "discover_task_warning",
                          "task": action.id, "dropped_dep": dep,
                          "reason": "unknown_dependency_id"})

        new_task = {
            "id": action.id,
            "title": action.title,
            "description": action.description,
            "status": "pending",
            "owner": None,
            "assigned_by": None,
            "priority": len(tasks["tasks"]) + 1,
            "dependencies": valid_deps,
            "files": [],
        }
        tasks["tasks"].append(new_task)
        self.log({"type": "discover_task", "agent": agent_name, "task": action.id,
                  "title": action.title, "dependencies": valid_deps})
        
        self._graph.add_node(action.id, **new_task)
        for dep in valid_deps:
            self._graph.add_edge(dep, action.id)

    def step_agent(self, agent_name, round_number, is_lead=False):
        agent = self.agents[agent_name]

        tasks = self.load_tasks()
        if is_lead:
            context = self._get_agent_context(agent_name, tasks, is_lead=True)
            agent.receive(context, sender="System")
        else:
            # Only inject full context when the agent is idle or has a new task assignment.
            # Active teammates already have their task description in history.
            # If the buffer would otherwise be empty, send a minimal nudge so the reply fires.
            last_shown = self._agent_last_shown_task.get(agent_name)
            current_task = agent.current_task
            if current_task is None or current_task != last_shown:
                context = self._get_agent_context(agent_name, tasks, is_lead=False)
                agent.receive(context, sender="System")
                self._agent_last_shown_task[agent_name] = current_task
            elif not agent.buffer:
                agent.receive(f"Continue working on task {current_task}.", sender="System")

        # Lead gets a fresh task summary every round it runs, so stale task summaries
        # in history are low-value. Use a tighter window for the lead.
        max_hist = 10 if is_lead else 20
        reply = agent.reply(max_history_messages=max_hist)
        self.log({"type": "agent_reply", "agent": agent_name, "round": round_number, "text": reply[:2000]})

        actions = parse_actions(reply)
        tasks = self._process_actions(agent_name, actions, tasks)
        self.save_tasks(tasks)

    async def step_agent_async(self, agent_name, round_number, is_lead=False):
        agent = self.agents[agent_name]

        tasks = self.load_tasks()
        if is_lead:
            context = self._get_agent_context(agent_name, tasks, is_lead=True)
            agent.receive(context, sender="System")
        else:
            # Only inject full context when the agent is idle or has a new task assignment.
            # Active teammates already have their task description in history.
            # If the buffer would otherwise be empty, send a minimal nudge so the reply fires.
            last_shown = self._agent_last_shown_task.get(agent_name)
            current_task = agent.current_task
            if current_task is None or current_task != last_shown:
                context = self._get_agent_context(agent_name, tasks, is_lead=False)
                agent.receive(context, sender="System")
                self._agent_last_shown_task[agent_name] = current_task
            elif not agent.buffer:
                agent.receive(f"Continue working on task {current_task}.", sender="System")

        # Lead gets a fresh task summary every round it runs, so stale task summaries
        # in history are low-value. Use a tighter window for the lead.
        max_hist = 10 if is_lead else 20
        reply = await agent.reply_async(max_history_messages=max_hist)
        self.log({"type": "agent_reply", "agent": agent_name, "round": round_number, "text": reply[:2000]})

        actions = parse_actions(reply)
        tasks = self.load_tasks()  
        tasks = self._process_actions(agent_name, actions, tasks)
        self.save_tasks(tasks)

    def initialize_agents(self, lead_agent_name=None):
        """Send initial context to all agents."""
        self._lead_agent_name = lead_agent_name
        tasks = self.load_tasks()
        success_criteria = tasks.get('success_criteria', 'Make all tests pass.')
        repository = tasks.get('repository', tasks.get('project', ''))
        project_intro = f"""
=== PROJECT: {tasks['project']} ===
{tasks['description']}

Success Criteria: {success_criteria}

Repository: {repository}
"""
        for agent in self.agents.values():
            agent.receive(project_intro, sender="System")
        teammate_names = [n for n in self.agents if n != lead_agent_name]
        self.log({
            "type": "initialize",
            "project": tasks["project"],
            "agents": list(self.agents.keys()),
            "num_teammates": len(teammate_names),
        })

    async def planning_phase(self, lead_agent_name, planning_prompt, max_planning_turns=5):
        """
        Run the planning phase: lead explores the codebase then emits a seed task graph.

        The lead may use <run_script>, <read_file>, and <run_tests> any number of times
        before emitting <discover_task> tags.  The loop continues until discover_task
        tags appear (or max_planning_turns is exhausted).
        """
        self._allowed_graph_ops = {DiscoverTaskAction}
        self._current_round = 0
        self.log({"type": "planning_phase_start", "agent": lead_agent_name})

        lead = self.agents[lead_agent_name]

        tasks = self.load_tasks()
        project_context = (
            f"=== PROJECT: {tasks['project']} ===\n\n"
            f"{tasks['description']}\n\n"
            f"Success criteria: {tasks.get('success_criteria', 'Complete all tasks successfully.')}\n"
        )
        lead.receive(planning_prompt, sender="System")
        lead.receive(project_context, sender="System")

        n_discovered = 0
        for turn in range(max_planning_turns):
            reply = await lead.reply_async()
            self.log({"type": "planning_reply", "agent": lead_agent_name,
                      "turn": turn, "text": reply[:4000]})
            actions = parse_actions(reply)

            # Handle exploration actions and feed results back
            has_exploration = False
            for a in actions:
                if isinstance(a, RunScriptAction):
                    has_exploration = True
                    returncode, output = self.run_script(a.path)
                    status = "✅ Script completed" if returncode == 0 else f"❌ Script exited with code {returncode}"
                    lead.receive(f"{status}:\n{output}", sender="System")
                elif isinstance(a, RunTestsAction):
                    has_exploration = True
                    self.run_tests()
                    lead.receive(f"Test output:\n{self._last_test_stdout[-2000:]}", sender="System")
                elif isinstance(a, ReadFileAction):
                    has_exploration = True
                    rel = a.path[len("repo/"):] if a.path.startswith("repo/") else a.path
                    abs_path = self.repo_dir / rel
                    content = abs_path.read_text()[:4000] if abs_path.exists() else "[file not found]"
                    lead.receive(f"=== {rel} ===\n{content}", sender="System")

            tasks = self.load_tasks()
            tasks = self._process_actions(lead_agent_name, actions, tasks)
            self.save_tasks(tasks)
            n_discovered = len(tasks["tasks"])

            if n_discovered > 0:
                self.log({"type": "planning_phase_end", "tasks_discovered": n_discovered,
                          "turns": turn + 1})
                break

            if not has_exploration:
                lead.receive(
                    "Your response contained no <discover_task> tags and no exploration actions. "
                    "Use <run_tests />, <run_script>, or <read_file> to explore, then emit "
                    "<discover_task> tags to seed the task graph.",
                    sender="System",
                )
        else:
            self.log({"type": "planning_phase_end", "tasks_discovered": n_discovered,
                      "turns": max_planning_turns, "reason": "max_turns"})
        
        # In static mode, lock the graph after planning — no mid-run discovery allowed.
        self._allowed_graph_ops = set() if self.static_graph else {DiscoverTaskAction}
        self._current_round = None

        if n_discovered == 0:
            raise RuntimeError(
                "Planning phase produced zero tasks after retry. "
                "The lead did not emit any <discover_task> tags. "
                "Check the planning prompt and project description."
            )

        # Validate the graph is a DAG (no cycles) 
        self._validate_dag(tasks["tasks"])

        return n_discovered

    def _validate_dag(self, _task_list=None):
        if not nx.is_directed_acyclic_graph(self._graph):
            cycle = nx.find_cycle(self._graph)
            raise RuntimeError(f"Cyclic dependency detected: {cycle}")

    def run(self, max_rounds=20, lead_agent_name=None):
        """Sequential run (debugging)."""
        self.log({"type": "run_start"})
        self.initialize_agents(lead_agent_name=lead_agent_name)

        for r in range(1, max_rounds + 1):
            self._current_round = r
            self._round_file_writers.clear()
            self.log({"type": "round_start", "round": r})

            if lead_agent_name and lead_agent_name in self.agents:
                self.step_agent(lead_agent_name, round_number=r, is_lead=True)

            for name in self.agents.keys():
                if name != lead_agent_name:
                    self.step_agent(name, round_number=r, is_lead=False)

            tasks = self.load_tasks()
            if all(t["status"] == "done" for t in tasks["tasks"]):
                if self._inject_fix_task_if_tests_failing(tasks, r):
                    continue 

                tests_passed = self.allow_run_tests and (self._last_test_returncode == 0)
                self.log({"type": "run_end", "reason": "all_tasks_done", "round": r,
                          "tests_passed": tests_passed})
                self._save_final_state()
                return tests_passed

        self.log({"type": "run_end", "reason": "max_rounds"})
        self._save_final_state()
        return False

    async def _close_async_clients(self):
        for agent in self.agents.values():
            client = getattr(agent, "llm_client", None)
            async_client = getattr(client, "async_client", None)
            if async_client is not None:
                close = getattr(async_client, "close", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result

    def _heartbeat_check(self, lead_agent_name: str | None, teammate_names: list) -> int:
        """
        Monitoring check — no LLM call.

        Detects agents that have been active on a task for >= _HEARTBEAT_STUCK_ROUNDS rounds
        without writing any file OR running any script.  Agents that are actively running
        scripts (e.g. exploring the filesystem, testing incrementally) are NOT considered
        stuck even if they haven't edited an implementation file yet.

        Returns the number of new alerts injected.
        """
        if lead_agent_name is None or lead_agent_name not in self.agents:
            return 0
        if self._current_round is None:
            return 0

        alerts_injected = 0
        for name in teammate_names:
            agent = self.agents[name]
            task_id = agent.current_task
            if task_id is None:
                continue
            if task_id in self._heartbeat_alerted_tasks:
                continue  # already alerted for this task

            assigned_at = self._task_assigned_round.get(task_id, self._current_round)
            rounds_active = self._current_round - assigned_at
            last_edit   = self._agent_last_edit_round.get(name, 0)
            last_script = self._agent_last_script_round.get(name, 0)
            last_activity = max(last_edit, last_script)
            rounds_without_activity = self._current_round - last_activity

            if (rounds_active >= self._HEARTBEAT_STUCK_ROUNDS
                    and rounds_without_activity >= self._HEARTBEAT_STUCK_ROUNDS):
                self._heartbeat_alerted_tasks.add(task_id)
                self.agents[lead_agent_name].receive(
                    f"⚠️ Heartbeat: {name} has been on task '{task_id}' for {rounds_active} rounds "
                    f"without editing any files or running any scripts (last activity: round {last_activity}). "
                    f"They may be stuck — check in with them or use "
                    f"<release_task id=\"{task_id}\" /> to reassign.",
                    sender="System",
                )
                self.log({
                    "type": "heartbeat_alert",
                    "agent": name,
                    "task": task_id,
                    "rounds_active": rounds_active,
                    "last_activity_round": last_activity,
                    "rounds_without_activity": rounds_without_activity,
                })
                alerts_injected += 1

        return alerts_injected

    async def run_async(self, max_rounds=20, lead_agent_name=None):
        """
        Async run: lead executes first each round, then all teammates run in parallel.
        """
        try:
            self.log({"type": "run_start"})
            self.initialize_agents(lead_agent_name=lead_agent_name)

            teammate_names = [n for n in self.agents if n != lead_agent_name]

            # Track state at last Lead invocation to decide when to skip.
            _lead_last_done_count  = -1   # tasks done when lead last ran
            _lead_last_graph_size  = -1   # graph size when lead last ran
            _lead_consecutive_skip = 0    # rounds skipped since last invocation
            _LEAD_MAX_SKIP = 6
            _consecutive_no_dispatch = 0  

            for r in range(1, max_rounds + 1):
                self._current_round = r
                self._round_file_writers.clear()
                self.log({"type": "round_start", "round": r})
                released = self._check_stragglers()

                # Heartbeat
                if lead_agent_name and lead_agent_name in self.agents:
                    self._heartbeat_check(lead_agent_name, teammate_names)

                # Lead goes first (sequential) so assignments are visible to teammates.
                # Skip the Lead when nothing has changed since it last ran.
                # The heartbeat fills the lead's buffer with alerts when agents are stuck,
                # so the lead is still invoked when needed — just not on empty rounds.
                if lead_agent_name and lead_agent_name in self.agents:
                    tasks_now      = self.load_tasks()
                    done_now       = sum(1 for t in tasks_now["tasks"] if t["status"] == "done")
                    graph_size_now = len(tasks_now["tasks"])
                    all_idle       = all(self.agents[n].current_task is None
                                        for n in teammate_names)
                    team_stuck     = all_idle and not self.get_ready_tasks()
                    something_new  = (
                        done_now       != _lead_last_done_count or
                        graph_size_now != _lead_last_graph_size or
                        released                                  # straggler(s) freed
                    )
                    lead_has_alerts = bool(self.agents[lead_agent_name].buffer)
                    invoke_lead = (
                        r == 1 or
                        something_new or
                        team_stuck or                             # all idle, nothing ready
                        lead_has_alerts or                        # heartbeat injected an alert
                        _lead_consecutive_skip >= _LEAD_MAX_SKIP
                    )
                    if invoke_lead:
                        await self.step_agent_async(lead_agent_name, round_number=r, is_lead=True)
                        _lead_last_done_count  = done_now
                        _lead_last_graph_size  = graph_size_now
                        _lead_consecutive_skip = 0
                    else:
                        _lead_consecutive_skip += 1
                        self.log({"type": "lead_skipped", "round": r,
                                  "reason": "no_change", "consecutive_skips": _lead_consecutive_skip})

                # Determine which teammates to dispatch this round.
                # Active agents always run.
                # Idle agents only run if there are ready tasks to claim.
                active = [n for n in teammate_names
                          if self.agents[n].current_task is not None]
                idle   = [n for n in teammate_names
                          if self.agents[n].current_task is None]

                dispatched_active = []
                for n in active:
                    has_new_content     = bool(self.agents[n].buffer)
                    was_just_dispatched = self._agent_last_dispatched.get(n, -1) == r - 1
                    if has_new_content or not was_just_dispatched:
                        dispatched_active.append(n)
                    else:
                        self.log({"type": "agent_skipped_no_new_context",
                                  "agent": n, "round": r})

                n_idle_slots = len(self.get_ready_tasks())
                to_dispatch  = dispatched_active + idle[:n_idle_slots]

                if not to_dispatch and idle and _consecutive_no_dispatch >= 1:
                    to_dispatch = idle[:1]

                self.log({"type": "round_dispatch",
                          "round": r,
                          "n_dispatched": len(to_dispatch),
                          "n_total": len(teammate_names),
                          "dynamic": self.dynamic_agents})

                if to_dispatch:
                    _consecutive_no_dispatch = 0
                else:
                    _consecutive_no_dispatch += 1

                for n in to_dispatch:
                    self._agent_last_dispatched[n] = r

                await asyncio.gather(*[
                    self.step_agent_async(name, round_number=r, is_lead=False)
                    for name in to_dispatch
                ])

                tasks = self.load_tasks()
                if all(t["status"] == "done" for t in tasks["tasks"]):
                    if self._inject_fix_task_if_tests_failing(tasks, r):
                        continue  
                    tests_passed = self.allow_run_tests and (self._last_test_returncode == 0)
                    self.log({"type": "run_end", "reason": "all_tasks_done", "round": r,
                              "tests_passed": tests_passed})
                    self._save_final_state()
                    return tests_passed

            self.log({"type": "run_end", "reason": "max_rounds"})
            self._save_final_state()
            return False
        finally:
            await self._close_async_clients()

    def _inject_fix_task_if_tests_failing(self, tasks: dict, round_number: int) -> bool:
        if not self.allow_run_tests:
            return False
        passed = self.run_tests()
        if passed:
            return False

        if self.static_graph:
            return False

        existing_ids = {t["id"] for t in tasks["tasks"]}
        fix_id = "fix-failing-tests"
        n = 1
        while fix_id in existing_ids:
            fix_id = f"fix-failing-tests-{n}"
            n += 1

        all_task_ids = [t["id"] for t in tasks["tasks"]]
        action = DiscoverTaskAction(
            id=fix_id,
            title="Fix failing tests",
            description=(
                "All planned tasks are marked done but the test suite is still failing.\n\n"
                f"Failing output:\n{self._last_test_stdout[-2000:]}\n\n"
                "Run <run_tests /> to see the current failures. "
                "Read the test suite and the current source files carefully. "
                "Fix every failing test, then run <run_tests /> again to confirm all pass. "
                "Use <discover_task> to add further tasks if the fix is large."
            ),
            dependencies=all_task_ids,
        )
        self._apply_discover_task(action, tasks, agent_name="System")
        self.save_tasks(tasks)

        failure_summary = self._last_test_stdout[-800:]
        self.broadcast_message(
            "System",
            f"⚠️ All tasks marked done but tests are still failing. "
            f"Task '{fix_id}' has been added to the graph.\n\n{failure_summary}",
        )
        self.log({"type": "fix_task_injected", "task": fix_id, "round": round_number})
        return True

    def _save_final_state(self):
        tasks = self.load_tasks()
        final_tasks_path = self.run_dir / "final_tasks.json"
        final_tasks_path.write_text(json.dumps(tasks, indent=2))
    
    def get_ready_tasks(self) -> list[str]:
        return [
            n for n in self._graph.nodes
            if self._graph.nodes[n]["status"] == "pending"
            and self._deps_satisfied(n)
        ]

    def critical_path(self) -> list[str]:
        return nx.dag_longest_path(self._graph)

    def unlocked_by(self, task_id: str) -> list[str]:
        return list(self._graph.successors(task_id))
