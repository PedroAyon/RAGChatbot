"""
QueryAgent: abstract base for agents that answer natural-language questions
against a DataSource.

An agent asks the LLM to translate the question into a backend-native query,
runs it through ``DataSource.execute_query`` and wraps the result in a
QueryResponse. If the LLM's reply can't be parsed or the query fails, the
error is appended to the prompt and the LLM is asked again, up to
``ATTEMPT_LIMIT`` times.

``query()`` implements that loop once. Subclasses supply only what is
backend-specific:

* ``_build_prompt``: the initial prompt (schema, dialect rules, examples).
* ``_describe_columns``: descriptions for the columns of a result.

The LLM must reply with JSON in one of two shapes (see
``_response_format_instructions``):

    {"can_answer": true, "<query_key>": "<generated query>"}
    {"can_answer": false, "reason": "<why>"}
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..data_source.base import ColumnarData, DataSource, Specification
from ..llm.base import BaseLLM
from ..models import QueryResponse

logger = logging.getLogger(__name__)

ATTEMPT_LIMIT = 6

_DEFAULT_REASON = "The query cannot be answered with the available data."
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class QueryAgent(ABC):
    query_key: str = "sql"
    """JSON key under which the LLM returns the generated query.
    Override in subclasses whose queries aren't SQL."""

    def __init__(self, llm: BaseLLM, data_source: DataSource):
        self.llm = llm
        self.data_source = data_source

    def get_overview(self) -> str:
        return self.data_source.get_overview()

    # -- public API -----------------------------------------------------------

    def query(self, natural_language_query: str) -> QueryResponse:
        """Answer a natural-language question about the data source.

        Returns a tabular QueryResponse on success, or a non-tabular one
        carrying an explanation if the LLM says the question can't be
        answered or no valid query was produced within ``ATTEMPT_LIMIT``
        attempts.
        """
        specification = self.data_source.get_specification()
        prompt = self._build_prompt(natural_language_query, specification)

        for attempt in range(1, ATTEMPT_LIMIT + 1):
            try:
                llm_response = self.llm.send(prompt)
                logger.debug("Attempt %d LLM response: %s", attempt, llm_response)
                parsed = self._parse_llm_json(llm_response.content)
                if not parsed.get("can_answer", False):
                    return QueryResponse(
                        response_text=parsed.get("reason", _DEFAULT_REASON),
                        is_tabular=False,
                    )

                generated_query = parsed.get(self.query_key)
                if not generated_query:
                    raise ValueError(f"No '{self.query_key}' key provided in the JSON response.")

                data = self.data_source.execute_query(generated_query)
            except Exception as exc:  # parse and backend errors are fed back to the LLM
                logger.warning("Attempt %d failed: %s", attempt, exc)
                prompt += self._retry_feedback(attempt, exc)
                continue

            return self._build_result(data, specification)

        return QueryResponse(
            response_text=f"Failed to generate and execute a valid query after {ATTEMPT_LIMIT} attempts.",
            is_tabular=False,
        )

    # -- hooks for subclasses ---------------------------------------------------

    @abstractmethod
    def _build_prompt(self, natural_language_query: str, specification: Specification) -> str:
        """Build the initial prompt: schema, dialect rules, the user's question
        and the expected JSON format (see ``_response_format_instructions``)."""

    @abstractmethod
    def _describe_columns(self, data: ColumnarData, specification: Specification) -> Dict[str, str]:
        """Map each column of `data` to a human-readable description ("" if unknown).

        Result columns may be aliases or aggregates (e.g. ``COUNT(*)``) that
        aren't in the specification.
        """

    # -- shared helpers -------------------------------------------------------------

    def _response_format_instructions(self, example_query: str) -> str:
        """The JSON-reply instructions block, for subclasses to put in their prompt.

        `example_query` is a short, backend-flavored placeholder shown to
        the LLM, e.g. ``SELECT ... FROM people ...``.
        """
        return (
            "=== INSTRUCTIONS ===\n"
            "If the user's query can be answered using this schema, output a JSON object like this:\n"
            f'{{"can_answer": true, "{self.query_key}": "{example_query}"}}\n\n'
            "If the query CANNOT be answered with this data (missing columns, unrelated topic), output:\n"
            '{"can_answer": false, "reason": "Explanation of why"}\n\n'
            "Respond ONLY with valid JSON. Do not include conversational text or markdown code blocks."
        )

    def _build_result(self, data: ColumnarData, specification: Specification) -> QueryResponse:
        row_count = max((len(column) for column in data.values()), default=0)
        text = "Executed query successfully."
        if row_count == 0:
            text = "Executed query successfully, but returned 0 rows (no data matched the criteria)."

        return QueryResponse(
            response_text=text,
            is_tabular=True,
            data=data,
            column_descriptions=self._describe_columns(data, specification),
        )

    @staticmethod
    def _retry_feedback(attempt: int, error: Exception) -> str:
        return (
            f"\n\nAttempt {attempt} failed.\n"
            f"Error: {error}\n"
            "Please analyze the error, fix the query if necessary, and return the corrected JSON object."
        )

    @staticmethod
    def _parse_llm_json(text: str) -> Dict[str, Any]:
        """Parse the LLM's reply as a JSON object, tolerating a Markdown code fence."""
        parsed = json.loads(_CODE_FENCE.sub("", text.strip()))
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
        return parsed