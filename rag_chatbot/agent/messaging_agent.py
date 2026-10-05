from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from pathlib import Path
from itertools import zip_longest
from typing import Any, Dict, List, Optional

from ..llm.base import BaseLLM
from ..models import Message, MessageHistory, MessageSource, QueryResponse, Tool
from ..query.base import QueryAgent

logger = logging.getLogger(__name__)

ColumnData = Dict[str, List[Any]]

# Above this many rows, a QueryResponse isn't entirely fed to the llm's context, just the data's columns.
_TABULAR_ROW_THRESHOLD = 10

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a database. "
    "Use the query_data_source tool to fetch data using natural language requests. "
    "If a result is large/tabular, prefer exporting it to CSV. "
    "You can use the charting tools to transform the tabular data into charts "
    "using only the column names. "
    "Don't share technical details with the user. "
    "Write your text responses in plain text, not markdown."
)

_CSV_SUCCESS = "Success: The CSV file has been generated and attached to the response."
_CHART_SUCCESS = "Success: The chart has been generated and attached to the response."


class MessagingAgent:
    def __init__(
            self,
            llm: BaseLLM,
            csv_output_dir: str = "./output/csv",
            chart_output_dir: str = "./output/charts",
            system_prompt: Optional[str] = None,
            query_agent: Optional[QueryAgent] = None,
            db_config_path: Optional[str] = None,
    ):
        self.llm = llm
        self.query_agent = query_agent
        self.db_config_path = Path(db_config_path) if db_config_path else None
        self.query_agent = query_agent
        self.csv_output_dir = csv_output_dir
        self.chart_output_dir = chart_output_dir
        self.llm.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        os.makedirs(self.csv_output_dir, exist_ok=True)
        os.makedirs(self.chart_output_dir, exist_ok=True)

        self._last_query_response: Optional[QueryResponse] = None
        self._pending_attachments: List[str] = []

        self._register_tools()

    # -- public API -------------------------------------------------------

    def send(self, message_history: MessageHistory) -> Message:
        """Entry point used by a MessagingService: run one conversational
        turn and return the agent's reply as a Message."""
        if not self.query_agent:
            return self._agent_message(message_history.user_id, "Please select a data source")

        self._pending_attachments = []
        llm_response = self.llm.send(self._build_prompt(message_history))

        csv_paths = [p for p in self._pending_attachments if p.lower().endswith(".csv")]
        image_paths = [p for p in self._pending_attachments if not p.lower().endswith(".csv")]

        return self._agent_message(
            message_history.user_id,
            llm_response.content,
            file_paths=csv_paths,
            image_paths=image_paths,
        )

    def set_query_agent(self, query_agent: QueryAgent) -> None:
        self.query_agent = query_agent
        self._last_query_response = None
        self._pending_attachments = []

    def connect(self, data_source_id: str) -> str:
        """Connect to one of the data sources listed in ``db_config_path``.

        Builds the data source's LLM + QueryAgent stack and attaches it to
        this agent, resetting the previous query state. Returns a user-facing
        status message. Failures are reported in it, not raised, and leave the
        current data source untouched.
        """
        # Imported lazily to avoid a circular import with the config package.
        from ..config import ComponentFactory

        if self.db_config_path is None:
            return "No data source configuration file is set."
        if not self.db_config_path.exists():
            return f"Configuration file missing at {self.db_config_path}"

        try:
            config = json.loads(self.db_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "Failed to parse the JSON configuration file."

        sources = config.get("data_sources", {})
        if data_source_id not in sources:
            available = ", ".join(sources) or "None configured"
            return f"Data source '{data_source_id}' not found.\nAvailable sources: {available}"

        try:
            self.set_query_agent(ComponentFactory.build_query_stack(sources[data_source_id]))
        except Exception as exc:
            logger.error("Error setting up data source '%s': %s", data_source_id, exc)
            return f"Error setting up data source '{data_source_id}': {exc}"

        return f"✅ Successfully connected to `{data_source_id}`! You can now query this database."

    # -- tools registered with the LLM ------------------------------------

    def query_data_source_tool(self, query: str) -> str:
        """Ask the QueryAgent a natural-language question about the data
        source. Returns a text description of the result: the direct
        answer for small results, or a schema summary (pointing at
        create_csv_file / the chart tools) for large tabular ones."""
        if not self.query_agent:
            raise ValueError("query_agent is required")
        response = self.query_agent.query(query)
        logger.debug("Query agent responded: %s", response)
        self._last_query_response = response

        if not response.is_tabular:
            return response.response_text

        if response.row_count > _TABULAR_ROW_THRESHOLD:
            return (
                f"The query matched {response.row_count} rows, too many to inline. "
                f"Available columns: {self._summarize_columns(response)}. "
                "Call create_csv_file to export the full result, or one of the "
                "chart tools to visualize it."
            )

        data_str = json.dumps(response.data, default=str)
        return f"{response.response_text}\n\nData:\n{data_str}"

    def create_csv_file(
            self, data: Optional[ColumnData] = None, columns: Optional[List[str]] = None
    ) -> str:
        """Export tabular data as a CSV file and return its path."""
        resolved = self._resolve_data(data, columns)
        path = os.path.join(self.csv_output_dir, f"export_{uuid.uuid4().hex}.csv")

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(resolved.keys())
            # Shorter columns are padded with "" so ragged data still exports.
            writer.writerows(zip_longest(*resolved.values(), fillvalue=""))

        self._pending_attachments.append(path)
        return _CSV_SUCCESS

    def column_chart_tool(
            self, data: Optional[ColumnData] = None, columns: Optional[List[str]] = None,
            title: str = "Column chart",
    ) -> str:
        resolved = self._resolve_data(data, columns)
        keys = list(resolved)
        if len(keys) < 2:
            raise ValueError("column_chart_tool needs at least two columns: one for labels, one (or more) for values")

        fig, ax = self._plt().subplots()
        x_key, *y_keys = keys
        x_data = resolved[x_key]

        # Evenly spaced numeric grid for the bars, labeled with the real x values.
        x_positions = range(len(x_data))
        for y_key in y_keys:
            ax.bar(x_positions, resolved[y_key], label=y_key)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_data, rotation=45, ha="right")

        return self._finish_xy_chart(fig, ax, x_key, y_keys, title)

    def line_chart_tool(
            self, data: Optional[ColumnData] = None, columns: Optional[List[str]] = None,
            title: str = "Line chart",
    ) -> str:
        resolved = self._resolve_data(data, columns)
        keys = list(resolved)
        if len(keys) < 2:
            raise ValueError(
                "line_chart_tool needs at least two columns: one for the x-axis, one (or more) for y series"
            )

        fig, ax = self._plt().subplots()
        x_key, *y_keys = keys

        # Let Matplotlib handle raw data types (dates/floats) on its own.
        for y_key in y_keys:
            ax.plot(resolved[x_key], resolved[y_key], label=y_key)
        self._plt().setp(ax.get_xticklabels(), rotation=45, ha="right")

        return self._finish_xy_chart(fig, ax, x_key, y_keys, title)

    def pie_chart_tool(
            self, data: Optional[ColumnData] = None, columns: Optional[List[str]] = None,
            title: str = "Pie chart",
    ) -> str:
        resolved = self._resolve_data(data, columns)
        if len(resolved) != 2:
            raise ValueError("pie_chart_tool needs exactly two columns: labels and values")

        label_col, value_col = resolved
        fig, ax = self._plt().subplots()
        ax.pie(resolved[value_col], labels=resolved[label_col], autopct="%1.1f%%")
        ax.set_title(title)
        return self._save_figure(fig)

    def histogram_tool(
            self, data: Optional[ColumnData] = None, columns: Optional[List[str]] = None,
            bins: int = 10, title: str = "Histogram",
    ) -> str:
        resolved = self._resolve_data(data, columns)
        if len(resolved) != 1:
            raise ValueError("histogram_tool needs exactly one numeric column")

        (column,) = resolved
        fig, ax = self._plt().subplots()
        ax.hist(resolved[column], bins=bins)
        ax.set_xlabel(column)
        ax.set_title(title)
        return self._save_figure(fig)

    # -- tool registration ----------------------------------------------------

    def _register_tools(self) -> None:
        def schema(required: List[str], **extra_props: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": {"columns": {"type": "array", "items": {"type": "string"}}, **extra_props},
                "required": required,
            }

        title = {"type": "string"}
        tools = [
            Tool(
                name="query_data_source",
                description="Query the connected data source with a natural-language question.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The natural-language question."}},
                    "required": ["query"],
                },
                callable=self.query_data_source_tool,
            ),
            Tool(
                name="create_csv_file",
                description="Export the most recent large query result (or a given set of columns from it) to a CSV file.",
                parameters=schema([]),
                callable=self.create_csv_file,
            ),
            Tool(
                name="column_chart_tool",
                description="Render a bar/column chart from named columns of the most recent query result (first column = labels).",
                parameters=schema(["columns"], title=dict(title)),
                callable=self.column_chart_tool,
            ),
            Tool(
                name="pie_chart_tool",
                description="Render a pie chart from exactly two named columns (labels, values) of the most recent query result.",
                parameters=schema(["columns"], title=dict(title)),
                callable=self.pie_chart_tool,
            ),
            Tool(
                name="line_chart_tool",
                description="Render a line chart from named columns of the most recent query result (first column = x-axis).",
                parameters=schema(["columns"], title=dict(title)),
                callable=self.line_chart_tool,
            ),
            Tool(
                name="histogram_tool",
                description="Render a histogram from exactly one named numeric column of the most recent query result.",
                parameters=schema(["columns"], bins={"type": "integer"}, title=dict(title)),
                callable=self.histogram_tool,
            ),
        ]
        for tool in tools:
            self.llm.register_tool(tool)

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _agent_message(user_id: str, text: str, **attachments: List[str]) -> Message:
        return Message(user_id=user_id, source=MessageSource.AGENT, message=text, **attachments)

    @staticmethod
    def _summarize_columns(response: QueryResponse) -> str:
        """Format columns as: 'Col1' (description), 'Col2'"""
        descriptions = response.column_descriptions or {}
        parts = []
        for col in response.data or {}:
            desc = descriptions.get(col)
            parts.append(f"'{col}' ({desc})" if desc else f"'{col}'")
        return ", ".join(parts)

    def _resolve_data(
            self, data: Optional[ColumnData], columns: Optional[List[str]]
    ) -> ColumnData:
        """Prefer explicit `data`; otherwise pull `columns` out of the last
        tabular QueryResponse -- this is what keeps large results out of
        the LLM's tool-call arguments."""
        if data:
            return data

        last = self._last_query_response
        last_data = last.data if last else None

        if columns:
            if not last_data:
                raise ValueError("No prior tabular query result to pull columns from.")
            missing = [c for c in columns if c not in last_data]
            if missing:
                raise ValueError(f"Unknown column(s): {missing}")
            return {c: last_data[c] for c in columns}

        if last_data:
            return last_data
        raise ValueError("No data available: pass 'data', pass 'columns', or run a tabular query first.")

    def _build_prompt(self, message_history: MessageHistory) -> str:
        lines = self._state_lines()

        for m in message_history.message_list:
            speaker = "User" if m.source == MessageSource.USER else "Assistant"
            lines.append(f"{speaker}: {m.message}")

        lines.append("Assistant:")
        return "\n".join(lines)

    def _state_lines(self) -> List[str]:
        """Inject the active state so the LLM remembers what it just queried
        and exactly which columns are available for follow-up charting."""
        last = self._last_query_response
        if not last:
            return []

        lines = [
            "System: [INTERNAL STATE] The last query executed returned:\n"
            f"{last.response_text}"
        ]
        if last.is_tabular:
            columns = list((last.data or {}).keys())
            lines.append(
                "System: [INTERNAL STATE] A tabular dataset is currently in memory "
                f"with {last.row_count} rows. "
                f"The exact column names available for chart/export tools are: {columns}. "
                "Use THESE exact column names when calling charting tools. "
                "This is the result of a past query. "
                "If the data available is not useful to your purpose, try querying again."
            )
        return lines

    @staticmethod
    def _plt():
        # Imported lazily so the package doesn't hard-require matplotlib
        # unless chart tools are actually used.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    def _finish_xy_chart(self, fig, ax, x_key: str, y_keys: List[str], title: str) -> str:
        ax.set_xlabel(x_key)
        ax.set_title(title)
        if len(y_keys) > 1:
            ax.legend()
        return self._save_figure(fig)

    def _save_figure(self, fig) -> str:
        path = os.path.join(self.chart_output_dir, f"chart_{uuid.uuid4().hex}.png")
        fig.savefig(path, bbox_inches="tight")
        self._plt().close(fig)
        self._pending_attachments.append(path)
        return _CHART_SUCCESS
