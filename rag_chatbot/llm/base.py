"""
BaseLLM: abstract base for any LLM provider (OpenAI, Ollama, ...).

Callers (MessagingAgent, QueryAgent) only ever call ``send()``, once per
turn. The tool-calling loop is *not* implemented here because it is
inherently provider-specific: each API represents tool calls and tool
results differently. Concrete providers implement that loop and use the
helpers below:

* ``register_tool`` / ``tools``: the tools the model may call.
* ``_execute_tool``: look up a tool by name and run it.
* ``system_prompt``: instructions sent ahead of every prompt. Agents set it
  on their LLM, so providers must send the *current* value on each request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import LLMResponse, Tool


class LLMTimeoutError(Exception):
    """The llm provider did not answer within its timeout."""


class BaseLLM(ABC):
    def __init__(
            self,
            model: str,
            source: str,
            tools: Optional[List[Tool]] = None,
            system_prompt: Optional[str] = None,
    ):
        """
        Args:
            model: Provider-specific model name (e.g. ``"gemma4:26b"``).
            source: Label identifying the provider (e.g. ``"ollama"``).
            tools: Tools to register up front. More can be added later with
                ``register_tool``.
            system_prompt: Optional system message. Agents may replace it
                after construction (``llm.system_prompt = ...``).
        """
        self.model = model
        self.source = source
        self.system_prompt = system_prompt
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register_tool(tool)

    # -- tools ----------------------------------------------------------------

    @property
    def tools(self) -> List[Tool]:
        """Registered tools, in registration order."""
        return list(self._tools.values())

    def register_tool(self, tool: Tool) -> None:
        """Make a Tool callable by this LLM.

        Registering a tool whose name is already taken replaces the old one.
        """
        self._tools[tool.name] = tool

    def _execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Run the registered tool called `tool_name` with `kwargs`.

        Raises:
            ValueError: if no tool with that name is registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(
                f"'{tool_name}' is not a registered tool for this LLM. "
                f"Available tools: {sorted(self._tools)}"
            )
        return tool.callable(**kwargs)

    # -- provider contract ------------------------------------------------------

    @abstractmethod
    def send(self, prompt: str) -> LLMResponse:
        """Send `prompt` to the model and return its final response, after
        running any tool calls the model requested along the way.

        Raises:
            LLMTimeoutError: if the provider doesn't answer in time.
        """
