import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class ParallelizationMetrics:

    def __init__(self, events_path, tasks_path):
        self.events_path = events_path
        self.tasks_path = tasks_path
        self.events = self._load_events()
        self.tasks = self._load_tasks()

    def _load_events(self):
        events = []
        with self.events_path.open() as f:
            for line in f:
                events.append(json.loads(line))
        return events

    def _load_tasks(self):
        return json.loads(self.tasks_path.read_text())

    def get_total_rounds(self):
        round_starts = [e for e in self.events if e["type"] == "round_start"]
        return len(round_starts)

    def get_task_assignments(self):
        assignments = defaultdict(list)

        for event in self.events:
            if event["type"] == "assign_task":
                assignments[event["assignee"]].append(event["task"])

        for event in self.events:
            if event["type"] == "claim_task":
                assignments[event["agent"]].append(event["task"])

        return dict(assignments)

    def get_file_lock_conflicts(self):
        conflicts = 0
        for event in self.events:
            if event["type"] == "lock_attempt" and not event["ok"]:
                conflicts += 1
        return conflicts

    def get_concurrent_writes(self):
        """Count rounds where 2+ agents wrote the same file."""
        writes: Dict[tuple, set] = defaultdict(set)
        for event in self.events:
            if event["type"] == "edit_file":
                key = (event.get("round", 0), event["path"])
                writes[key].add(event["agent"])
        return sum(1 for agents in writes.values() if len(agents) > 1)

    def get_claims_blocked(self):
        """Count times an agent tried to claim a task whose dependencies weren't done."""
        return sum(1 for e in self.events if e["type"] == "claim_blocked")

    def get_agent_activity(self):
        activity = defaultdict(lambda: defaultdict(int))

        for event in self.events:
            if event["type"] in ["claim_task", "complete_task", "edit_file"]:
                agent = event["agent"]
                round_num = event.get("round", 0)
                activity[agent][round_num] += 1

        return {agent: dict(rounds) for agent, rounds in activity.items()}

    def get_task_completion_order(self):
        completions = []
        for event in self.events:
            if event["type"] == "complete_task":
                completions.append({
                    "task_id": event["task"],
                    "agent": event["agent"],
                    "round": event.get("round", 0),
                    "timestamp": event["ts"]
                })
        return completions

    @staticmethod
    def compute_critical_path(tasks_list):
        deps = {t["id"]: t.get("dependencies", []) for t in tasks_list}
        memo: Dict[str, int] = {}

        def longest_path(task_id: str) -> int:
            if task_id in memo:
                return memo[task_id]
            predecessors = deps.get(task_id, [])
            if not predecessors:
                memo[task_id] = 1
            else:
                memo[task_id] = 1 + max(longest_path(p) for p in predecessors)
            return memo[task_id]

        if not tasks_list:
            return 0
        return max(longest_path(t["id"]) for t in tasks_list)

    @staticmethod
    def amdahl_speedup(p, n):
        if n <= 0:
            return 1.0
        return 1.0 / ((1.0 - p) + p / n)

    def get_parallelization_score(self):
        assignments = self.get_task_assignments()
        total_rounds = self.get_total_rounds()
        num_tasks = len(self.tasks["tasks"])

        init_event = next((e for e in self.events if e["type"] == "initialize"), None)
        if init_event and "num_teammates" in init_event:
            num_teammates = init_event["num_teammates"]
            num_agents = num_teammates + 1 
        else:
            agent_names = {
                e["agent"] for e in self.events
                if e.get("type") == "agent_reply" and e.get("agent")
            }
            num_teammates = len(agent_names - {"Lead"})
            num_agents = len(agent_names)

        agents_used = sum(1 for tasks in assignments.values() if tasks)

        if agents_used > 0:
            avg_tasks_per_agent = num_tasks / agents_used
            task_distribution_variance = sum(
                (len(tasks) - avg_tasks_per_agent) ** 2
                for tasks in assignments.values()
            ) / agents_used
        else:
            avg_tasks_per_agent = 0
            task_distribution_variance = 0

        critical_path = self.compute_critical_path(self.tasks.get("tasks", []))
        workers = max(1, num_teammates)
        theoretical_min_rounds = max(critical_path, math.ceil(num_tasks / workers))
        p_parallelizable = (num_tasks - critical_path) / num_tasks if num_tasks > 0 else 0.0

        predicted_speedup = self.amdahl_speedup(p_parallelizable, workers)

        if total_rounds > 0:
            efficiency = theoretical_min_rounds / total_rounds
        else:
            efficiency = 0

        return {
            "total_rounds": total_rounds,
            "num_tasks": num_tasks,
            "num_teammates": num_teammates,
            "num_agents": num_agents,
            "agents_used": agents_used,
            "avg_tasks_per_agent": avg_tasks_per_agent,
            "task_distribution_variance": task_distribution_variance,
            "critical_path": critical_path,
            "p_parallelizable": p_parallelizable,
            "theoretical_min_rounds": theoretical_min_rounds,
            "predicted_amdahl_speedup": predicted_speedup,
            "efficiency": efficiency,
            "lock_conflicts": self.get_file_lock_conflicts(),
            "concurrent_writes": self.get_concurrent_writes(),
            "claims_blocked": self.get_claims_blocked(),
        }

    def generate_report(self) -> str:
        assignments = self.get_task_assignments()
        score = self.get_parallelization_score()
        completions = self.get_task_completion_order()

        report = []
        report.append("=" * 70)
        report.append("PARALLELIZATION METRICS REPORT")
        report.append("=" * 70)
        report.append("")

        report.append(f"Project: {self.tasks['project']}")
        report.append(f"Total Tasks: {score['num_tasks']}")
        report.append(f"Teammates (workers): {score['num_teammates']}")
        report.append(f"Total Agents (incl. lead): {score['num_agents']}")
        report.append(f"Agents Used: {score['agents_used']}")
        report.append(f"Critical Path Length: {score['critical_path']} rounds")
        report.append(f"Parallelizability (p): {score['p_parallelizable']:.2f}")
        report.append("")

        report.append("Task Assignments:")
        for agent, tasks in sorted(assignments.items()):
            report.append(f"  {agent}: {tasks}")
        report.append("")

        report.append(f"Completion Order:")
        for completion in completions:
            report.append(f"  Round {completion['round']}: {completion['task_id']} by {completion['agent']}")
        report.append("")

        report.append("Efficiency Metrics:")
        report.append(f"  Total Rounds: {score['total_rounds']}")
        report.append(f"  Theoretical Min Rounds: {score['theoretical_min_rounds']} (max of critical path and ceil(tasks/workers))")
        report.append(f"  Predicted Amdahl Speedup (vs n=1): {score['predicted_amdahl_speedup']:.2f}x")
        report.append(f"  Efficiency: {score['efficiency']:.2%}")
        report.append(f"  Avg Tasks/Agent: {score['avg_tasks_per_agent']:.2f}")
        report.append(f"  Task Distribution Variance: {score['task_distribution_variance']:.2f}")
        report.append(f"  File Lock Conflicts: {score['lock_conflicts']}")
        report.append("")

        # Analysis
        report.append("Analysis:")
        if score['efficiency'] > 0.8:
            report.append("High parallelization efficiency")
        elif score['efficiency'] > 0.5:
            report.append("Moderate parallelization efficiency")
        else:
            report.append("Low parallelization efficiency")

        if score['task_distribution_variance'] < 1.0:
            report.append("Evenly distributed workload")
        else:
            report.append("Uneven workload distribution")

        if score['lock_conflicts'] == 0:
            report.append("No file lock conflicts")
        elif score['lock_conflicts'] < 5:
            report.append("Some file lock conflicts")
        else:
            report.append("Many file lock conflicts")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


def analyze_run(run_dir: Path) -> ParallelizationMetrics:
    events_path = run_dir / "events.jsonl"
    tasks_path = run_dir / "final_tasks.json"
    return ParallelizationMetrics(events_path, tasks_path)
