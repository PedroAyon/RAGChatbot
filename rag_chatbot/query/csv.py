"""
CSVQueryAgent: natural language -> DuckDB SQL over a CSVDataSource.

Everything backend-agnostic (retry loop, JSON handling, response building)
lives in QueryAgent. This class contributes the DuckDB-specific prompt and
the column descriptions for the CSV's single table.
"""

from __future__ import annotations

import json
from typing import Dict

from .base import QueryAgent
from ..data_source import CSVDataSource
from ..data_source.base import ColumnarData, Specification
from ..llm import BaseLLM

_DUCKDB_CHEAT_SHEET = """\
=== DUCKDB SQL CHEAT SHEET ===
You are querying a DuckDB backend. Use DuckDB-specific functions when helpful:
Use TRY_CAST(col AS type) instead of CAST() to gracefully handle malformed CSV rows (returns NULL on failure).
Use STRPTIME(col, '%Y-%m-%d') to parse custom date strings.
Use DATE_TRUNC('month', date_col) for time-series grouping.
Use REGEXP_EXTRACT(col, 'pattern', 1) to extract regex capture groups.
Use STRING_SPLIT(col, ',') to turn delimited strings into arrays.
Use CONTAINS(col, 'text') instead of LIKE for simple substring searches.
Use ILIKE for case-insensitive matching.

To bin continuous numeric or float columns into groups, use DuckDB integer division (//) like this: (col // n) * n, or use FLOOR(col / n) * n."""


class CSVQueryAgent(QueryAgent):
    data_source: CSVDataSource

    def __init__(self, llm: BaseLLM, data_source: CSVDataSource):
        if not isinstance(data_source, CSVDataSource):
            raise TypeError(
                f"CSVQueryAgent requires a CSVDataSource, got {type(data_source).__name__}"
            )
        super().__init__(llm, data_source)

    def _build_prompt(self, natural_language_query: str, specification: Specification) -> str:
        table_name = self.data_source.table_name
        sections = [
            "You are an expert DuckDB SQL data analyst.\n"
            "Your task is to translate a natural language query into a valid, read-only SQL query.",
            f"=== DATABASE CONTEXT ===\nOverview:\n{self.get_overview()}",
            "Schema (Table -> Column -> Type & Description):\n"
            f"{json.dumps(specification, indent=2)}",
            self._response_format_instructions(f"SELECT ... FROM {table_name} ..."),
            f"User Query: {natural_language_query}",
            _DUCKDB_CHEAT_SHEET,
        ]
        return "\n\n".join(sections)

    def _describe_columns(self, data: ColumnarData, specification: Specification) -> Dict[str, str]:
        table_columns = specification.get(self.data_source.table_name, {})
        return {
            col: (table_columns.get(col) or {}).get("description") or ""
            for col in data
        }