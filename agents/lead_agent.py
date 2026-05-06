from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from agents.base_agent import BaseAgent


@dataclass
class LeadAgent(BaseAgent):

    team_members: List[str] = field(default_factory=list)

    def __init__(self, name, system_prompt, model=None, team_members=None, static_graph=False):
        super().__init__(name, system_prompt, model)
        self.team_members = team_members or []
        self.system_prompt = self._build_lead_prompt(system_prompt, static_graph=static_graph)

    def _build_lead_prompt(self, base_prompt: str, static_graph: bool = False) -> str:
        """Inject runtime team composition into the base prompt."""
        team_line = f"\nYou are coordinating {len(self.team_members)} teammates: {', '.join(self.team_members)}.\n"
        return base_prompt + team_line

    def _idle_response(self):
        """Lead agent can check on team status when idle."""
        return "<request_status />"
