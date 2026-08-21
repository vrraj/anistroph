"""Search configuration models — searchable fields and semantic filters.

Parsed from the ``search:`` section of a dataset.yaml file. The search
engine (filters.py, service.py) consults these to validate queries and
expand semantic filters into deterministic predicates.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchFieldSpec(BaseModel):
    """A single searchable field.

    Attributes:
        field: the underlying dataset column name.
        operators: supported operators for this field (subset of
            eq, in, gte, lte, between, contains_range).
        unit: optional unit label (e.g. "MT/s", "Gb", "C") shown in the
            search contract and used for display.
        aliases: alternative names an agent might use (e.g. "speed" for
            data_rate_mt_s). The contract lists these so the agent can
            map natural-language terms to the canonical field.
        description: human-readable description shown in the contract.
    """

    field: str
    operators: list[str] = Field(default_factory=lambda: ["eq", "in"])
    unit: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None


class SemanticFilterSpec(BaseModel):
    """A named semantic filter that expands to deterministic predicates.

    Two types:
        range_contains — maps a single value to ``min_field <= value AND
            max_field >= value`` (e.g. "supports 55C" over operating temp).
        expands_to — maps to an explicit list of FilterExpression dicts
            (e.g. "industrial_temperature" -> min <= -40 AND max >= 95).
    """

    name: str
    type: str  # "range_contains" | "expands_to"
    # For range_contains:
    min_field: Optional[str] = None
    max_field: Optional[str] = None
    # For expands_to: list of {field, op, value} dicts
    expands_to: list[dict[str, Any]] = Field(default_factory=list)
    unit: Optional[str] = None
    description: Optional[str] = None


class SearchConfig(BaseModel):
    """Full search configuration for a dataset."""

    searchable_fields: dict[str, SearchFieldSpec] = Field(default_factory=dict)
    semantic_filters: dict[str, SemanticFilterSpec] = Field(default_factory=dict)

    def field_for_alias(self, alias: str) -> Optional[str]:
        """Return the canonical field name for an alias, or None."""
        for name, spec in self.searchable_fields.items():
            if alias == name or alias in spec.aliases:
                return spec.field
        return None

    def semantic_filter(self, name: str) -> Optional[SemanticFilterSpec]:
        return self.semantic_filters.get(name)
