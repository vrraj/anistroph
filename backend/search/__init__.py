"""Generic parametric search — structured filtering over registered datasets.

This package is domain-agnostic. A dataset declares its searchable fields,
units, aliases, and semantic filters in a ``search:`` section of its
dataset.yaml. The engine applies deterministic Polars predicates; no
memory-specific logic lives here.
"""
