import asyncio
import time
import os
from typing import Optional
from anthropic import Anthropic, AsyncAnthropic
from openai import OpenAI, AsyncOpenAI

# Retry config for transient API errors
_MAX_ATTEMPTS  = 6
_BASE_DELAY    = 2.0   
_MAX_DELAY     = 60.0

class LLMClient:

    def __init__(self, provider, model):
        """
        Initialize LLM client.

        Args:
            provider: "anthropic", "openai"
            model: Model name
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._initialize_clients()

    def _get_default_model(self):
        """Get default model."""
        defaults = {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
        }
        return defaults.get(self.provider, "claude-sonnet-4-6")

    def _initialize_clients(self):
        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in .env file")
            self.client = Anthropic(api_key=api_key)
            self.async_client = AsyncAnthropic(api_key=api_key)

        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported: anthropic, openai.")

    _USES_MAX_COMPLETION_TOKENS = ("o1", "o3", "gpt-5")

    def _openai_tokens_kwarg(self, max_tokens: int) -> dict:
        if any(self.model.startswith(p) for p in self._USES_MAX_COMPLETION_TOKENS):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def call(self, system, messages, temperature=0.7, max_tokens=4096):
        """
        Call the LLM with system prompt and messages.
        """
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            return response.content[0].text
        elif self.provider == "openai":
            full_messages = [{"role": "system", "content": system}] + messages
            tokens_kwarg = self._openai_tokens_kwarg(max_tokens)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                **tokens_kwarg,
            )
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content

    async def call_async(self, system, messages, temperature=0.7, max_tokens=4096):
        if self.provider == "anthropic":
            response = await self.async_client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            return response.content[0].text
        elif self.provider == "openai":
            full_messages = [{"role": "system", "content": system}] + messages
            tokens_kwarg = self._openai_tokens_kwarg(max_tokens)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                **tokens_kwarg,
            )
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content

def create_llm_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    model = model or os.getenv("LLM_MODEL")
    return LLMClient(provider=provider, model=model)
