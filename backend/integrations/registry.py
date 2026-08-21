"""External tool registry — loads and validates external tool definitions.

Reads ``integrations/tool_registry.yaml`` and produces ``ExternalToolDef``
objects. The MCP server and REST API consume the same registry instance
rather than duplicating external-tool definitions.

Environment variable substitution is supported in ``base_url`` and ``path``
fields using ``${VAR_NAME}`` syntax (e.g. ``${VERIS_BASE_URL}``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "integrations" / "tool_registry.yaml"

_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _substitute_env(value: str) -> str:
    """Replace ${VAR_NAME} with the corresponding environment variable."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return _VAR_PATTERN.sub(_replace, value)


class ExternalToolDef(BaseModel):
    """A single externally-hosted tool definition."""

    name: str
    provider: str
    capability: str
    visibility: str = "always"  # "always" | "mcp_only" | "rest_only" | "hidden"
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    llm_parameters: dict[str, Any] = Field(default_factory=dict)
    agent_owner: str = ""
    protocol: str = "A2A_JSONRPC"
    base_url: str = ""
    path: str = ""

    @field_validator("base_url", "path")
    @classmethod
    def _substitute_env_vars(cls, v: str) -> str:
        return _substitute_env(v)

    @property
    def is_mcp_visible(self) -> bool:
        """Whether this tool should be exposed through MCP."""
        return self.visibility in ("always", "mcp_only")

    @property
    def is_rest_visible(self) -> bool:
        """Whether this tool should be exposed through REST."""
        return self.visibility in ("always", "rest_only")

    @property
    def resolved_url(self) -> str:
        """Full URL = base_url + path."""
        base = self.base_url.rstrip("/")
        path = self.path if self.path.startswith("/") else "/" + self.path
        return base + path


class ExternalToolRegistry:
    """Loads and holds external tool definitions from YAML."""

    def __init__(self, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.registry_path = Path(registry_path)
        self._tools: dict[str, ExternalToolDef] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self._tools = {}
            return
        with open(self.registry_path) as f:
            data = yaml.safe_load(f) or {}
        tools = data.get("tools", [])
        for t in tools:
            tool = ExternalToolDef(**t)
            self._tools[tool.name] = tool

    def reload(self) -> None:
        """Re-read the registry from disk."""
        self._tools.clear()
        self._load()

    def get(self, name: str) -> Optional[ExternalToolDef]:
        return self._tools.get(name)

    def list_all(self) -> list[ExternalToolDef]:
        return list(self._tools.values())

    def list_mcp_visible(self) -> list[ExternalToolDef]:
        return [t for t in self._tools.values() if t.is_mcp_visible]

    def list_rest_visible(self) -> list[ExternalToolDef]:
        return [t for t in self._tools.values() if t.is_rest_visible]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Module-level singleton — loaded once, reused by MCP and REST.
_registry: Optional[ExternalToolRegistry] = None


def get_external_tool_registry() -> ExternalToolRegistry:
    """Return the shared external tool registry singleton."""
    global _registry
    if _registry is None:
        _registry = ExternalToolRegistry()
    return _registry


def reset_external_tool_registry() -> None:
    """Reset the singleton (for testing)."""
    global _registry
    _registry = None
