from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from agents.agent_interface import AgentInterface
from agents.llm_client import create_llm_client, LLMClient
from config import get_config


@dataclass
class BaseAgent(AgentInterface):
    """Base agent with common functionality for all agent types."""

    name: str
    system_prompt: str
    model: Optional[dict] = None
    buffer: List[Dict[str, str]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    llm_client: Optional[LLMClient] = None

    # Agent state
    current_task: Optional[str] = None
    is_idle: bool = True

    def __init__(self, name, system_prompt, model=None):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model or {}
        self.buffer = []
        self.history = []
        self.current_task = None
        self.is_idle = True

        # Initialize LLM client
        config = get_config()
        provider = self.model.get("provider", config.llm.provider)
        model_name = self.model.get("model", config.llm.model)
        self.llm_client = create_llm_client(provider=provider, model=model_name)

        # Store LLM parameters
        self.temperature = self.model.get("temperature", config.llm.temperature)
        self.max_tokens = self.model.get("max_tokens", config.llm.max_tokens)

    def receive(self, message, sender=None):
        """Receive a message from another agent or the system."""
        sender_label = sender or "System"
        self.buffer.append({
            "role": "user",
            "content": f"[From {sender_label}]: {message}"
        })

    def reply(self, max_history_messages=20):
        """Generate reply using LLM and clear buffer.

        Args:
            max_history_messages: Keep only the last N messages in history
        """
        if not self.buffer:
            # If no messages in buffer, agent is idle
            return self._idle_response()

        # Truncate history to avoid unbounded context growth
        recent_history = self.history[-max_history_messages:]

        # Call LLM with recent conversation history + new messages
        response = self.llm_client.call(
            system=self.system_prompt,
            messages=recent_history + self.buffer,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Move buffer to history
        self.history.extend(self.buffer)
        self.history.append({"role": "assistant", "content": response})
        self.buffer = []

        return response

    async def reply_async(self, max_history_messages=20):
        """Async version of reply(). Used for parallel agent execution."""
        if not self.buffer:
            return self._idle_response()

        recent_history = self.history[-max_history_messages:]

        response = await self.llm_client.call_async(
            system=self.system_prompt,
            messages=recent_history + self.buffer,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        self.history.extend(self.buffer)
        self.history.append({"role": "assistant", "content": response})
        self.buffer = []

        return response

    def _idle_response(self):
        """What the agent does when idle with no messages."""
        return ""

    def get_context_summary(self):
        """Get a summary of current context for status updates."""
        task_info = f"Current task: {self.current_task}" if self.current_task else "No current task"
        status = "Idle" if self.is_idle else "Working"
        return f"{self.name} - {status} - {task_info}"

    def claim_task(self, task_id: str):
        """Mark a task as claimed by this agent."""
        self.current_task = task_id
        self.is_idle = False

    def complete_task(self):
        """Mark current task as complete and reset history for a clean slate on the next task."""
        self.current_task = None
        self.is_idle = True
        self.history = []

    def get_status(self) -> Dict[str, any]:
        """Get agent status for orchestrator."""
        return {
            "name": self.name,
            "is_idle": self.is_idle,
            "current_task": self.current_task,
            "message_count": len(self.buffer),
            "history_length": len(self.history)
        }
