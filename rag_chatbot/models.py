from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MessageSource(Enum):
    """Who produced a given Message."""

    AGENT = "agent"
    USER = "user"


@dataclass
class Message:
    """A single chat message, going either user -> agent or agent -> user.

    file_path / image_path are optional attachments (e.g. a generated CSV
    export or a generated chart) that a MessagingService should forward to
    the underlying chat client when present.
    """

    user_id: str
    source: MessageSource
    message: str
    file_paths: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)


@dataclass
class MessageHistory:
    """The running conversation for a single user."""

    user_id: str
    message_list: List[Message] = field(default_factory=list)

    def add(self, message: Message) -> "MessageHistory":
        if message.user_id != self.user_id:
            raise ValueError(
                f"Message user_id '{message.user_id}' does not match "
                f"history user_id '{self.user_id}'"
            )
        self.message_list.append(message)
        return self

    def as_role_content_pairs(self) -> List[Dict[str, str]]:
        """Convenience conversion used by BaseLLM implementations that need
        a role/content list (OpenAI-style chat format) instead of the raw
        Message objects."""
        role_map = {MessageSource.USER: "user", MessageSource.AGENT: "assistant"}
        return [{"role": role_map[m.source], "content": m.message} for m in self.message_list]


@dataclass
class QueryResponse:
    """The result of a natural-language query against a DataSource.

    Two shapes are supported:

    1. Scalar / small answer: `data` is None (or trivially small) and
       `response_text` already contains the natural-language answer that
       can be fed straight back into the LLM's context.

    2. Tabular answer: `data` holds the result as columnar data
       (``{column_name: [values...]}``) together with `column_descriptions`
       so the agent can decide what to do with it without the raw rows
       ever needing to be pasted into the LLM's context window.
    """

    response_text: str
    is_tabular: bool = False
    data: Optional[Dict[str, List[Any]]] = None
    column_descriptions: Optional[Dict[str, str]] = None

    @property
    def row_count(self) -> int:
        if not self.data:
            return 0
        return max((len(values) for values in self.data.values()), default=0)


@dataclass
class LLMResponse:
    """What a BaseLLM.send() call returns once its internal tool-calling
    loop (if any) has finished."""

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class Tool:
    """A function the LLM is allowed to invoke."""

    name: str
    description: str
    parameters: Dict[str, Any]
    callable: Callable[..., Any]

    def to_schema(self) -> Dict[str, Any]:
        """JSON-schema-ish function definition, suitable for passing to most
        LLM function-calling APIs (OpenAI, Anthropic, Ollama, ...)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
