from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.llm_client import create_llm_client
from orchestrator.locks import LockManager
from orchestrator.token_parser import (
    parse_actions,
    EditFileAction, RunTestsAction, RunScriptAction, BroadcastAction,
)


class _NoGraphAgent(BaseAgent):

    def __init__(self, name: str, system_prompt: str, model: dict | None = None):
        super().__init__(name, system_prompt, model or {})

    def _idle_response(self) -> str:
        return ""


class NoGraphOrchestrator:

    def __init__(self, run_dir: Path, repo_dir: Path, agents: dict, task_description: str, allow_run_tests: bool = False):
        self.run_dir = run_dir
        self.repo_dir = repo_dir
        self.agents = agents
        self.task_description = task_description
        self.allow_run_tests = allow_run_tests
        self.lock_mgr = LockManager(run_dir / ".locks")
        self.events_path = run_dir / "events.jsonl"
        self._current_round: int | None = None
        self._last_test_stdout: str = ""
        self._lead_agent_name: str | None = None

    def log(self, event: dict):
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

    def apply_edit(self, agent_name: str, action: EditFileAction):
        rel = action.path
        if rel.startswith("repo/"):
            rel = rel[len("repo/"):]

        stem = Path(rel).stem
        if stem in self._STDLIB_SHADOWS:
            self.log({"type": "edit_blocked", "agent": agent_name, "path": rel,
                      "reason": "stdlib_shadow"})
            return False, "STDLIB_SHADOW"

        abs_path = self.repo_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        content = action.content
        content = re.sub(r'^```[^\n]*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        abs_path.write_text(content)
        self.log({"type": "edit_file", "agent": agent_name, "path": rel, "n_chars": len(content)})
        return True, "OK"

    def run_tests(self) -> bool:
        t0 = time.time()
        proc = subprocess.run(["pytest", "-q"], cwd=self.repo_dir, capture_output=True, text=True)
        dt = time.time() - t0
        self._last_test_stdout = proc.stdout
        self.log({
            "type": "run_tests",
            "returncode": proc.returncode,
            "seconds": dt,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-500:],
        })
        return proc.returncode == 0

    def run_script(self, path: str) -> tuple[int, str]:
        t0 = time.time()
        proc = subprocess.run(["python", path], cwd=self.repo_dir, capture_output=True, text=True)
        dt = time.time() - t0
        self.log({
            "type": "run_script",
            "path": path,
            "returncode": proc.returncode,
            "seconds": dt,
            "stdout": proc.stdout[-2000:],
        })
        output = proc.stdout[-3000:] + (f"\n[stderr]\n{proc.stderr[-1000:]}" if proc.stderr.strip() else "")
        return proc.returncode, output
    
    def broadcast_message(self, sender: str, message: str):
        for name, agent in self.agents.items():
            if name != sender:
                agent.receive(message, sender=sender)
        self.log({"type": "broadcast", "sender": sender, "message": message[:500]})

    def _process_actions(self, agent_name: str, actions: list):
        agent = self.agents[agent_name]
        # Deduplicate
        seen, deduped = set(), []
        for a in actions:
            key = (type(a).__name__,) + tuple(
                tuple(v) if isinstance(v, list) else v
                for v in vars(a).values()
            )
            if key not in seen:
                seen.add(key)
                deduped.append(a)

        for a in deduped:
            if isinstance(a, EditFileAction):
                self.apply_edit(agent_name, a)

            elif isinstance(a, RunTestsAction):
                if not self.allow_run_tests:
                    agent.receive(
                        "⚠️ <run_tests /> is not available in this experiment. "
                        "Write your own verification scripts with <edit_file> and run them with <run_script>.",
                        sender="System",
                    )
                else:
                    passed = self.run_tests()
                    msg = "✅ Tests passed!" if passed else f"❌ Tests failed:\n{self._last_test_stdout[-1500:]}"
                    agent.receive(msg, sender="System")

            elif isinstance(a, RunScriptAction):
                rc, output = self.run_script(a.path)
                status = "✅ Script completed" if rc == 0 else f"❌ Script exited with code {rc}"
                agent.receive(f"{status}:\n{output}", sender="System")

            elif isinstance(a, BroadcastAction):
                self.broadcast_message(agent_name, a.message)

    async def step_agent_async(self, agent_name: str, is_lead: bool = False):
        agent = self.agents[agent_name]

        if self._last_test_stdout:
            agent.receive(
                f"=== CURRENT TEST STATUS (round {self._current_round}) ===\n"
                f"{self._last_test_stdout[-1500:]}",
                sender="System",
            )

        reply = await agent.reply_async()
        self.log({
            "type": "agent_reply",
            "agent": agent_name,
            "round": self._current_round,
            "text": reply[:2000],
        })

        actions = parse_actions(reply)
        self._process_actions(agent_name, actions)

    def initialize_agents(self, lead_agent_name: str | None = None):
        self._lead_agent_name = lead_agent_name
        intro = f"=== PROJECT ===\n\n{self.task_description}"
        for agent in self.agents.values():
            agent.receive(intro, sender="System")
        self.log({
            "type": "initialize",
            "agents": list(self.agents.keys()),
            "lead": lead_agent_name,
        })

    async def run_async(self, max_rounds: int, lead_agent_name: str | None = None) -> bool:
        try:
            self.log({"type": "run_start"})
            all_names = list(self.agents.keys())
            peer_names = [n for n in all_names if n != lead_agent_name]

            for r in range(1, max_rounds + 1):
                self._current_round = r
                self.log({"type": "round_start", "round": r})

                if lead_agent_name and lead_agent_name in self.agents:
                    await self.step_agent_async(lead_agent_name, is_lead=True)

                # All peers run in parallel
                self.log({
                    "type": "round_dispatch",
                    "round": r,
                    "n_dispatched": len(peer_names),
                    "n_total": len(all_names),
                    "dynamic": False,
                })
                await asyncio.gather(*[
                    self.step_agent_async(name, is_lead=False)
                    for name in peer_names
                ])

                if self.run_tests():
                    self.log({"type": "run_end", "reason": "tests_passed", "round": r})
                    return True

            self.log({"type": "run_end", "reason": "max_rounds"})
            return False

        finally:
            await self._close_async_clients()

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

def _parse_pytest_counts(stdout: str) -> tuple[int, int]:
    """
    Parse pytest -q output for pass/fail counts.
    Returns (n_passed, n_total).  Falls back to (0, 0) if unparseable.
    """
    import re as _re

    passed = 0
    failed = 0
    m = _re.search(r'(\d+)\s+passed', stdout)
    if m:
        passed = int(m.group(1))
    m = _re.search(r'(\d+)\s+failed', stdout)
    if m:
        failed = int(m.group(1))
    return passed, passed + failed


async def run_no_graph_pipeline(
    task_file,
    n_teammates: int,
    max_rounds: int,
    run_dir: Path,
    model_config: dict | None = None,
    verbose: bool = True,
    allow_run_tests: bool = False,
) -> dict:
    task_file = Path(task_file)
    task_dir = task_file.parent

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".locks").mkdir(exist_ok=True)
    repo_dir = run_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("test_suite.py",):
        src = task_dir / fname
        if src.exists():
            shutil.copy(src, repo_dir / fname)

    setup_script = task_dir / "setup_data.py"
    if setup_script.exists():
        import importlib.util, sys as _sys
        _inserted = str(task_dir) not in _sys.path
        if _inserted:
            _sys.path.insert(0, str(task_dir))
        try:
            spec = importlib.util.spec_from_file_location("setup_data", setup_script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "setup"):
                mod.setup(repo_dir)
        finally:
            if _inserted:
                _sys.path.remove(str(task_dir))

    # Load prompts
    prompts_dir = Path("agents/prompts")
    lead_prompt     = (prompts_dir / "no_graph_lead_prompt.txt").read_text()
    teammate_prompt = (prompts_dir / "no_graph_teammate_prompt.txt").read_text()

    # Build agents
    teammate_names = [f"Dev{i+1}" for i in range(n_teammates)]
    agents: dict[str, _NoGraphAgent] = {
        "Lead": _NoGraphAgent(name="Lead", system_prompt=lead_prompt, model=model_config),
    }
    for name in teammate_names:
        agents[name] = _NoGraphAgent(name=name, system_prompt=teammate_prompt, model=model_config)

    # Load project description for agent init
    task_data = json.loads(task_file.read_text())
    task_description = task_data.get("description", "")

    orchestrator = NoGraphOrchestrator(
        run_dir=run_dir,
        repo_dir=repo_dir,
        agents=agents,
        task_description=task_description,
        allow_run_tests=allow_run_tests,
    )

    if verbose:
        print(f"  [no_graph] 1 lead + {n_teammates} teammates — no task graph")

    orchestrator.initialize_agents(lead_agent_name="Lead")

    t0 = time.time()
    success = await orchestrator.run_async(max_rounds=max_rounds, lead_agent_name="Lead")
    wall_clock = time.time() - t0

    total_input, total_output = 0, 0
    for agent in agents.values():
        client = getattr(agent, "llm_client", None)
        if client:
            total_input  += client.total_input_tokens
            total_output += client.total_output_tokens

    n_passed, n_total = _parse_pytest_counts(orchestrator._last_test_stdout)

    return {
        "success":              success,
        "tasks_done":           n_passed,
        "total_tasks":          n_total if n_total > 0 else 1,
        "wall_clock_seconds":   wall_clock,
        "total_input_tokens":   total_input,
        "total_output_tokens":  total_output,
        "run_dir":              str(run_dir),
        "n_teammates":          n_teammates,
        "task_file":            str(task_file),
    }


async def run_decentralized_pipeline(
    task_file,
    n_agents: int,
    max_rounds: int,
    run_dir: Path,
    model_config: dict | None = None,
    verbose: bool = True,
    allow_run_tests: bool = False,
) -> dict:
    """
    Fully decentralized condition: N peer agents with no designated leader.
    All agents run in parallel every round and self-organize via broadcast.
    No task graph, no leader/follower hierarchy.
    """
    task_file = Path(task_file)
    task_dir = task_file.parent

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".locks").mkdir(exist_ok=True)
    repo_dir = run_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("test_suite.py",):
        src = task_dir / fname
        if src.exists():
            shutil.copy(src, repo_dir / fname)

    setup_script = task_dir / "setup_data.py"
    if setup_script.exists():
        import importlib.util, sys as _sys
        _inserted = str(task_dir) not in _sys.path
        if _inserted:
            _sys.path.insert(0, str(task_dir))
        try:
            spec = importlib.util.spec_from_file_location("setup_data", setup_script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "setup"):
                mod.setup(repo_dir)
        finally:
            if _inserted:
                _sys.path.remove(str(task_dir))

    peer_prompt = (Path("agents/prompts") / "decentralized_prompt.txt").read_text()

    agent_names = [f"Dev{i+1}" for i in range(n_agents)]
    agents: dict[str, _NoGraphAgent] = {
        name: _NoGraphAgent(name=name, system_prompt=peer_prompt, model=model_config)
        for name in agent_names
    }

    task_data = json.loads(task_file.read_text())
    task_description = task_data.get("description", "")

    orchestrator = NoGraphOrchestrator(
        run_dir=run_dir,
        repo_dir=repo_dir,
        agents=agents,
        task_description=task_description,
        allow_run_tests=allow_run_tests,
    )

    if verbose:
        print(f"  [decentralized] {n_agents} peer agents — no leader, no task graph")

    # No lead: pass lead_agent_name=None so all agents run in parallel
    orchestrator.initialize_agents(lead_agent_name=None)

    t0 = time.time()
    success = await orchestrator.run_async(max_rounds=max_rounds, lead_agent_name=None)
    wall_clock = time.time() - t0

    total_input, total_output = 0, 0
    for agent in agents.values():
        client = getattr(agent, "llm_client", None)
        if client:
            total_input  += client.total_input_tokens
            total_output += client.total_output_tokens

    n_passed, n_total = _parse_pytest_counts(orchestrator._last_test_stdout)

    return {
        "success":              success,
        "tasks_done":           n_passed,
        "total_tasks":          n_total if n_total > 0 else 1,
        "wall_clock_seconds":   wall_clock,
        "total_input_tokens":   total_input,
        "total_output_tokens":  total_output,
        "run_dir":              str(run_dir),
        "n_agents":             n_agents,
        "task_file":            str(task_file),
    }
