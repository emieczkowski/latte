from __future__ import annotations
from dataclasses import dataclass
from agents.base_agent import BaseAgent


@dataclass
class TeammateAgent(BaseAgent):

    def __init__(self, name, system_prompt, model=None):
        super().__init__(name, system_prompt, model)
        self.system_prompt = self._build_teammate_prompt(system_prompt)

    def _build_teammate_prompt(self, base_prompt: str) -> str:
        """Augment prompt with teammate-specific instructions."""
        teammate_instructions = f"""
{base_prompt}

=== TEAMMATE AGENT ROLE ===

You are a developer on a team working collaboratively on coding tasks.

=== AVAILABLE ACTIONS ===

**Self-claim an unassigned task (when idle):**
<claim_task id="task-1" />

**Complete your current task:**
<complete_task id="task-1" />

**Edit a file:**
<edit_file path="relative/path.py">
file content here
</edit_file>

**Run tests:**
<run_tests />

**Propose a new subtask (when you discover work missing from the graph):**
<discover_task id="task-N" title="Short title" dependencies="task-A,task-B">
Full description of what the new subtask requires.
</discover_task>
Only emit this when you genuinely find that a necessary piece of work has no corresponding task yet.

=== BEHAVIOR ===

1. **When assigned a task**: Focus on completing it efficiently
2. **When idle**: Look for unassigned tasks and claim one
3. **When blocked**: Ask the lead for help or clarification
4. **File locking**: Files are automatically locked when you edit them
5. **Communication**: Be concise and report progress clearly

=== TASK WORKFLOW ===

1. Claim or receive task assignment
2. Understand requirements
3. Make necessary code changes
4. Run tests if applicable
5. Complete the task
6. You MUST emit <complete_task id="task-X" /> when done — without this the task stays pending and blocks the whole team.

Always use the action tags shown above. Keep responses focused on the task.
"""
        return teammate_instructions

    def _idle_response(self):
        """When idle with no messages, teammate can look for work to claim."""
        return "" # Orchestrator will provide task list.
