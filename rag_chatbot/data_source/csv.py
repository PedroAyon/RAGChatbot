"""
CSVDataSource: exposes a single CSV file as a queryable SQL table via DuckDB
(https://duckdb.org/docs/data/csv/overview).

DuckDB reads the CSV directly and runs full SQL against it, with no import
step and no database server. The file is registered as a view named
``table_name`` (default ``"data"``), so a CSV source always has exactly one
table.

Specification file
------------------
``specification_path`` is an optional YAML map of column descriptions::

    age: Patient age in years at admission
    sbp: Systolic blood pressure (mmHg)

Columns missing from the file get a description of None. DuckDB's inferred
column types are merged in by ``get_specification()``.

Credentials
-----------
``credentials`` is accepted to satisfy the DataSource contract but unused for
a local file. It exists so a remote CSV can carry auth later without changing the constructor.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

from .base import ColumnarData, DataSource, Specification

if TYPE_CHECKING:
    import duckdb

_READ_ONLY_PATTERN = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class CSVDataSource(DataSource):
    def __init__(
        self,
        overview_path: str,
        specification_path: str = "",
        csv_path: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        table_name: str = "data",
        csv_read_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            overview_path: Text file describing the data (see DataSource).
            specification_path: Optional YAML ``{column: description}`` map.
            csv_path: Path to the CSV file. Required (it is optional in the
                signature only because it is filled from a config dict).
            credentials: Unused for local files.
            table_name: Name of the SQL view the CSV is exposed as. Must be a
                plain identifier because it is interpolated into SQL.
            csv_read_options: Extra keyword options for DuckDB's
                ``read_csv_auto`` (e.g. ``{"delim": ";", "header": True}``).
        """
        if not csv_path:
            raise ValueError("CSVDataSource requires a 'csv_path' pointing at the CSV file to query.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: '{csv_path}'")

        super().__init__(overview_path, specification_path, credentials)
        self.csv_path = csv_path
        self.table_name = _validate_identifier(table_name)
        self.csv_read_options = csv_read_options or {}
        self._connection: Optional["duckdb.DuckDBPyConnection"] = None

    # -- DataSource contract ----------------------------------------------------

    def connect(self) -> bool:
        """Open an in-memory DuckDB connection and register the CSV as a view.

        Returns False (instead of raising) if the CSV cannot be read, so a
        bad or malformed file is a normal failure for the caller. Any
        previous connection is closed first.
        """
        duckdb = self._duckdb()
        self.close()

        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                f"CREATE OR REPLACE VIEW {self.table_name} AS "
                f"SELECT * FROM read_csv_auto('{_escape(self.csv_path)}'{self._read_csv_options_sql()})"
            )
            # Touch the view so a malformed CSV fails here, during connect(),
            # rather than on the first real query.
            connection.execute(f"SELECT * FROM {self.table_name} LIMIT 0")
        except duckdb.Error:
            connection.close()
            return False

        self._connection = connection
        return True

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def execute_query(self, query: str) -> ColumnarData:
        """Run a read-only SQL query and return ``{column: [values...]}``.

        This is the shape QueryResponse.data and MessagingAgent's chart/CSV
        tools expect. It is the primitive a QueryAgent calls after
        translating a question into SQL grounded in ``get_specification()``.

        Only SELECT / WITH (CTE) statements are accepted.

        Raises:
            ValueError: if the statement is not a SELECT/WITH.
        """
        if not _READ_ONLY_PATTERN.match(query):
            raise ValueError("Only read-only SELECT/WITH queries are allowed against a CSVDataSource.")

        cursor = self._require_connection().execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return {col: [row[i] for row in rows] for i, col in enumerate(columns)}

    def get_specification(self) -> Specification:
        """Return ``{table_name: {column: {"type", "description"}}}``.

        A CSV has a single table, so the result has one entry keyed by
        ``self.table_name``. Types are inferred by DuckDB; descriptions come
        from the specification file (None where absent).
        """
        # DESCRIBE yields (column_name, column_type, null, key, default, extra)
        rows = self._require_connection().execute(f"DESCRIBE {self.table_name}").fetchall()
        descriptions = self._load_column_descriptions()
        return {
            self.table_name: {
                name: {"type": col_type, "description": descriptions.get(name)}
                for name, col_type, *_ in rows
            }
        }

    # -- CSV-specific helpers -----------------------------------------------------

    def row_count(self) -> int:
        """Number of data rows in the CSV."""
        return self._require_connection().execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]

    # -- internals ------------------------------------------------------------------

    def _require_connection(self) -> "duckdb.DuckDBPyConnection":
        """Return the live connection, connecting lazily if needed."""
        if self._connection is None and not self.connect():
            raise RuntimeError(f"Could not connect to CSV data source at '{self.csv_path}'")
        return self._connection

    @staticmethod
    def _duckdb():
        # Imported lazily so the package doesn't hard-require duckdb unless a
        # CSVDataSource is used (same approach as matplotlib in MessagingAgent).
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "CSVDataSource requires the 'duckdb' package. Install it with "
                "`pip install duckdb` (it's listed in requirements.txt)."
            ) from exc
        return duckdb

    def _load_column_descriptions(self) -> Dict[str, str]:
        """Read the YAML ``{column: description}`` map, or {} if unavailable."""
        if not self.specification_path or not os.path.exists(self.specification_path):
            return {}
        with open(self.specification_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}

    def _read_csv_options_sql(self) -> str:
        """Render ``csv_read_options`` as extra ``read_csv_auto`` arguments."""
        return "".join(f", {key}={_sql_literal(value)}" for key, value in self.csv_read_options.items())


# -- module-level SQL helpers ------------------------------------------------------

def _escape(value: str) -> str:
    """Escape single quotes for use inside a SQL string literal."""
    return value.replace("'", "''")


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal (bool, number, or quoted string)."""
    if isinstance(value, bool):  # must precede int: bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{_escape(str(value))}'"


def _validate_identifier(name: str) -> str:
    """Restrict `name` to a plain SQL identifier.

    DuckDB can't bind identifiers as parameters, so table_name is
    interpolated into SQL and must be validated to prevent injection via config.
    """
    if not _IDENTIFIER_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid table_name '{name}': must be a valid SQL identifier")
    return name