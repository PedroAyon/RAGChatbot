from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Type, Union

import yaml

# Base classes
from ..data_source.base import DataSource
from ..llm.base import BaseLLM
from ..query.base import QueryAgent
from ..messaging.base import MessagingService
from ..agent.messaging_agent import MessagingAgent

# Concrete implementations
from ..llm.ollama import OllamaLLM
from ..data_source.csv import CSVDataSource
from ..query.csv import CSVQueryAgent
from ..messaging.telegram import TelegramMessagingService

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _resolve_env(value: Any, path: str = "config") -> Any:
    """
    Recursively replace ``${NAME}`` references in strings with the value of
    the environment variable NAME. Dicts and lists are traversed; other
    types are returned unchanged.

    Values taken from the environment are inserted as-is and never scanned
    again, so a secret that happens to contain ``${...}`` is safe.

    Raises:
        ValueError: if a referenced variable is not set. `path` says where
            in the config the reference was found.
    """
    if isinstance(value, str):
        def substitute(match: re.Match) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Environment variable '{name}' (referenced by '{path}') is not set.")
            return os.environ[name]

        return _ENV_REFERENCE.sub(substitute, value)
    if isinstance(value, dict):
        return {key: _resolve_env(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return value


def _lookup(registry: Dict[str, Type], kind: str, key: Any) -> Type:
    """Return registry[key] or raise a helpful error."""
    if not key or key not in registry:
        raise ValueError(f"Unknown or missing {kind} type: '{key}'. Registered: {list(registry)}")
    return registry[key]


class ComponentFactory:
    MESSAGING_SERVICE_REGISTRY: Dict[str, Type[MessagingService]] = {
        "telegram": TelegramMessagingService,
    }

    LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {
        "ollama": OllamaLLM,
    }

    DATA_SOURCE_REGISTRY: Dict[str, Type[DataSource]] = {
        "csv": CSVDataSource,
    }

    QUERY_AGENT_REGISTRY: Dict[str, Type[QueryAgent]] = {
        "csv": CSVQueryAgent,
    }

    # -- application entry point ---------------------------------------

    @classmethod
    def build_messaging_service(cls, yaml_path: Union[str, Path]) -> MessagingService:
        """Build the full chat stack (LLM -> MessagingAgent -> MessagingService)
        from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

        for section in ("messaging_agent", "messaging_service"):
            if not config.get(section):
                raise ValueError(f"'{yaml_path}' is missing the '{section}' block.")

        messaging_agent = cls.create_messaging_agent(config["messaging_agent"])
        return cls.create_messaging_service(config["messaging_service"], messaging_agent)

    # -- messaging -----------------------------------------------------

    @classmethod
    def create_messaging_agent(cls, config: Dict[str, Any]) -> MessagingAgent:
        raw = dict(config)
        llm_config = raw.pop("llm", None)
        if not llm_config:
            raise ValueError("Messaging agent configuration is missing the 'llm' block.")

        llm = cls.create_llm(llm_config)
        return MessagingAgent(llm=llm, **_resolve_env(raw, "messaging_agent"))

    @classmethod
    def create_messaging_service(
        cls, config: Dict[str, Any], messaging_agent: MessagingAgent
    ) -> MessagingService:
        config_copy = _resolve_env(config, "messaging_service")
        service_cls = _lookup(
            cls.MESSAGING_SERVICE_REGISTRY, "MessagingService", config_copy.pop("type", None)
        )
        return service_cls(messaging_agent=messaging_agent, **config_copy)

    # -- query stack ---------------------------------------------------

    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> BaseLLM:
        config_copy = _resolve_env(config, "llm")
        llm_cls = _lookup(cls.LLM_REGISTRY, "LLM", config_copy.pop("type", None))
        return llm_cls(**config_copy)

    @classmethod
    def create_data_source(cls, config: Dict[str, Any]) -> DataSource:
        # Keys that belong to the QueryAgent/LLM are dropped *before* resolving,
        # so the nested llm block isn't resolved a second time here.
        own_keys = {k: v for k, v in config.items() if k not in ("llm", "query_agent_type")}
        config_copy = _resolve_env(own_keys, "data_source")
        ds_cls = _lookup(cls.DATA_SOURCE_REGISTRY, "DataSource", config_copy.pop("type", None))
        return ds_cls(**config_copy)

    @classmethod
    def build_query_stack(cls, ds_config: Dict[str, Any]) -> QueryAgent:
        """Builds and connects the full stack: LLM + DataSource -> QueryAgent."""
        llm_config = ds_config.get("llm")
        if not llm_config:
            raise ValueError("Data source configuration is missing the 'llm' block.")

        llm = cls.create_llm(llm_config)
        data_source = cls.create_data_source(ds_config)

        if not data_source.connect():
            raise RuntimeError(f"Failed to connect to data source '{ds_config.get('type')}'.")

        # Fall back to the data source type if no query_agent_type is given
        qa_type = ds_config.get("query_agent_type", ds_config.get("type"))
        qa_cls = _lookup(cls.QUERY_AGENT_REGISTRY, "QueryAgent", qa_type)

        return qa_cls(llm=llm, data_source=data_source)