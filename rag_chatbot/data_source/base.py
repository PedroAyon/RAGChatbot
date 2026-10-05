"""
DataSource: abstract base for any underlying data backend
(CSV, PostgreSQL, MongoDB, ...).

A DataSource owns the connection to a backend and exposes four things to a
QueryAgent:

* ``connect`` / ``close``: connection lifecycle.
* ``get_specification``: the *shape* of the data (tables, columns, types,
  descriptions), used to ground natural-language-to-query translation.
* ``get_overview``: free-form, human-written context about the data.
* ``execute_query``: the low-level primitive that runs an already-translated
  query. DataSources do no natural-language understanding themselves.

Constructor arguments
---------------------
overview_path:
    Path to a text file with a high-level description of the data.
specification_path:
    Path to a backend-specific file with human-written column (and table)
    descriptions. Empty string means "none". Each backend documents the
    format it expects.
credentials:
    Whatever connection info the backend needs (host, user, password, URI,
    API key, ...). Backends that need none (e.g. a local file) ignore it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict


class ColumnSpec(TypedDict):
    """Description of a single column."""

    type: str
    """Backend-native type name (e.g. a DuckDB or PostgreSQL type)."""

    description: Optional[str]
    """Human-written meaning of the column, or None if undocumented."""


Specification = Dict[str, Dict[str, ColumnSpec]]
"""``{table_name: {column_name: ColumnSpec}}``.

Single-table backends (CSV) return exactly one entry. Multi-table backends
(PostgreSQL) return one entry per table, e.g. ``{"patients": {...}, "visits": {...}}``.
"""

ColumnarData = Dict[str, List[Any]]
"""Query results in columnar form: ``{column_name: [value, ...]}``."""


class DataSource(ABC):
    def __init__(
        self,
        overview_path: str,
        specification_path: str = "",
        credentials: Optional[Dict[str, Any]] = None,
    ):
        self.overview_path = overview_path
        self.specification_path = specification_path
        self.credentials: Dict[str, Any] = credentials or {}

    # -- lifecycle ------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Establish (or verify) the connection to the backend.

        Returns True on success and False on an expected failure (e.g. an
        unreadable file or unreachable server) rather than raising, so
        callers can report it as a normal error.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the connection and any resources. Safe to call repeatedly."""

    # -- data access ------------------------------------------------------------

    @abstractmethod
    def execute_query(self, query: str) -> ColumnarData:
        """Run an already-translated, read-only query and return columnar data.

        Implementations must reject anything that would modify the backend.
        """

    # -- metadata -----------------------------------------------------------------

    @abstractmethod
    def get_specification(self) -> Specification:
        """Describe the shape of the data as ``{table: {column: ColumnSpec}}``.

        Types come from the backend itself. Descriptions come from
        ``specification_path`` and are None wherever the file is silent.
        """

    def get_overview(self) -> str:
        """Return the contents of ``overview_path``, or "" if no path was
        given or the file does not exist."""
        if not self.overview_path or not os.path.exists(self.overview_path):
            return ""
        with open(self.overview_path, "r", encoding="utf-8") as fh:
            return fh.read()