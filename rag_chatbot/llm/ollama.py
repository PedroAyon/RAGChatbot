"""
OllamaLLM: BaseLLM backed by an Ollama server's chat API
(https://github.com/ollama/ollama/blob/main/docs/api.md#chat).

Implements the tool-calling loop that BaseLLM leaves to each provider:

1. Send the conversation, advertising every registered Tool.
2. If the reply contains ``tool_calls``, run each one through
   ``_execute_tool`` and append the results as ``tool`` messages.
3. Repeat until the model answers in plain text.

``max_tool_iterations`` caps the loop as a safety net against a model that
never stops calling tools. When it is hit, the model is asked one last time,
with no tools offered, to answer using what it already has.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from ..models import LLMResponse, Tool
from .base import BaseLLM, LLMTimeoutError

logger = logging.getLogger(__name__)

OllamaMessage = Dict[str, Any]  # {"role": ..., "content": ..., ...}


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        source: str = "ollama",
        tools: Optional[List[Tool]] = None,
        base_url: str = "http://localhost:11434",
        temperature: Optional[float] = None,
        max_tool_iterations: int = 10,
        request_timeout: int = 300,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            model: Ollama model name, e.g. ``"gemma4:26b"``.
            source: Provider label (see BaseLLM).
            tools: Tools to register up front.
            base_url: Ollama server URL.
            temperature: Sampling temperature; None uses the model's default.
            max_tool_iterations: Max chat round trips that may end in tool calls.
            request_timeout: Seconds to wait for each HTTP request. This is
                per chat round trip, not per ``send()``. The first request
                after Ollama starts loads the model and can be slow.
            system_prompt: Optional system message sent before every prompt
                (see BaseLLM; agents may replace it).
        """
        super().__init__(model=model, source=source, tools=tools, system_prompt=system_prompt)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tool_iterations = max_tool_iterations
        self.request_timeout = request_timeout

    # -- BaseLLM contract -------------------------------------------------------

    def send(self, prompt: str) -> LLMResponse:
        messages = self._build_messages(prompt)

        for _ in range(self.max_tool_iterations):
            reply = self._chat(messages)
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:
                return self._to_response(reply)

            messages.append(reply)
            messages.extend(self._run_tool_call(call) for call in tool_calls)

        logger.warning(
            "Model still requesting tools after %d iterations; forcing a final answer.",
            self.max_tool_iterations,
        )
        return self._to_response(self._chat(messages, force_no_tools=True))

    # -- Ollama API -----------------------------------------------------------------

    def _build_messages(self, prompt: str) -> List[OllamaMessage]:
        messages: List[OllamaMessage] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _chat(self, messages: List[OllamaMessage], force_no_tools: bool = False) -> OllamaMessage:
        """POST the conversation to ``/api/chat`` and return the reply message.

        Raises:
            LLMTimeoutError: if Ollama doesn't respond within ``request_timeout``.
        """
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if self.tools and not force_no_tools:
            payload["tools"] = [self._tool_to_ollama_schema(tool) for tool in self.tools]
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.request_timeout)
        except requests.Timeout as exc:
            raise LLMTimeoutError(
                f"Ollama did not respond within {self.request_timeout}s (model '{self.model}')."
            ) from exc
        resp.raise_for_status()
        return resp.json().get("message") or {}

    @staticmethod
    def _tool_to_ollama_schema(tool: Tool) -> Dict[str, Any]:
        return {"type": "function", "function": tool.to_schema()}

    @staticmethod
    def _to_response(message: OllamaMessage) -> LLMResponse:
        return LLMResponse(content=message.get("content", ""), tool_calls=None)

    # -- tool calls --------------------------------------------------------------------

    def _run_tool_call(self, call: Dict[str, Any]) -> OllamaMessage:
        """Execute one tool call and return the ``tool`` message carrying its result.

        Errors are reported back to the model as the tool's result instead of
        being raised, so one bad call doesn't crash the whole turn.
        """
        function = call.get("function", {})
        name = function.get("name") or ""
        arguments = self._parse_arguments(name, function.get("arguments"))
        logger.debug("Tool call: %s(%s)", name, arguments)

        try:
            result = self._execute_tool(name, **arguments)
        except LLMTimeoutError:
            raise  # an LLM timed out: abort the turn
        except Exception as exc:
            logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, exc)
            result = f"Error running tool '{name}': {exc}"

        return {"role": "tool", "name": name, "content": self._stringify(result)}

    @staticmethod
    def _parse_arguments(name: str, arguments: Any) -> Dict[str, Any]:
        """Normalize tool-call arguments to a dict.

        Ollama usually returns them already parsed, but some models emit a
        JSON string. Unparseable arguments become ``{}``.
        """
        if not arguments:
            return {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Could not parse tool arguments for '%s': %r", name, arguments)
                return {}
        return arguments

    @staticmethod
    def _stringify(value: Any) -> str:
        """Render a tool result as the string Ollama expects in ``content``."""
        return value if isinstance(value, str) else json.dumps(value, default=str)