"""Unit tests for the generic parametric search engine.

Tests the filter operators, semantic-filter expansion, and search service
against an in-memory Polars DataFrame registered as a test dataset.
"""

from __future__ import annotations

import pytest
import polars as pl

from backend.search.filters import (
    FilterExpression,
    Operator,
    SortExpression,
    apply_filter,
    apply_filters,
    expand_semantic,
    from_simple_dict,
)
from backend.search.spec import SearchConfig, SearchFieldSpec, SemanticFilterSpec
from backend.search.service import get_search_contract, search_dataset
from backend.datasets.registry import DatasetMeta, DatasetRegistry


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pl.DataFrame:
    """A small synthetic catalog-like DataFrame for search tests."""
    return pl.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4", "P5"],
        "product_family": ["DDR5_COMPONENT", "DDR5_COMPONENT", "LPDDR5X", "DDR5_RDIMM", "DDR5_COMPONENT"],
        "data_rate_mt_s": [4800, 6400, 8533, 6400, 7200],
        "bus_width_bits": [8, 8, 32, 80, 16],
        "component_density_gb": [16, 24, 64, None, 32],
        "operating_temp_min_c": [-40, 0, -40, 0, -40],
        "operating_temp_max_c": [95, 95, 105, 95, 95],
        "part_status": ["Production", "Production", "Sampling", "Production", "End of Life"],
    })


@pytest.fixture
def search_config() -> SearchConfig:
    return SearchConfig(
        searchable_fields={
            "product_family": SearchFieldSpec(
                field="product_family", operators=["eq", "in"],
                description="Memory product family.",
            ),
            "data_rate_mt_s": SearchFieldSpec(
                field="data_rate_mt_s", operators=["eq", "in", "gte", "lte", "between"],
                unit="MT/s", aliases=["speed"],
            ),
            "bus_width_bits": SearchFieldSpec(
                field="bus_width_bits", operators=["eq", "in", "gte", "lte"],
                unit="bits",
            ),
            "component_density_gb": SearchFieldSpec(
                field="component_density_gb", operators=["eq", "gte", "lte", "between"],
                unit="Gb",
            ),
            "part_status": SearchFieldSpec(
                field="part_status", operators=["eq", "in"],
            ),
        },
        semantic_filters={
            "operating_temperature": SemanticFilterSpec(
                name="operating_temperature",
                type="range_contains",
                min_field="operating_temp_min_c",
                max_field="operating_temp_max_c",
                unit="C",
                description="Products that support operation at a given temperature.",
            ),
            "industrial_temperature": SemanticFilterSpec(
                name="industrial_temperature",
                type="expands_to",
                description="Industrial temperature range (-40C to +95C).",
                expands_to=[
                    {"field": "operating_temp_min_c", "op": "lte", "value": -40},
                    {"field": "operating_temp_max_c", "op": "gte", "value": 95},
                ],
            ),
        },
    )


@pytest.fixture
def temp_registry(tmp_path, sample_df) -> DatasetRegistry:
    """Register a test dataset in a temp registry."""
    pq_path = tmp_path / "test_catalog.parquet"
    sample_df.write_parquet(str(pq_path))
    reg_path = tmp_path / "dataset_registry.json"
    reg = DatasetRegistry(reg_path)
    meta = DatasetMeta(
        dataset_id="test_catalog",
        name="Test Catalog",
        source="test",
        row_count=sample_df.height,
        columns=list(sample_df.columns),
        entity_key="product_id",
        parquet_path=str(pq_path),
    )
    reg._datasets["test_catalog"] = meta
    reg._save()
    return reg


# ---------------------------------------------------------------------------
# Operator tests
# ---------------------------------------------------------------------------

class TestOperators:
    def test_eq(self, sample_df):
        f = FilterExpression(field="product_family", op="eq", value="LPDDR5X")
        result = apply_filter(sample_df, f)
        assert result.height == 1
        assert result["product_id"][0] == "P3"

    def test_in(self, sample_df):
        f = FilterExpression(field="product_family", op="in", value=["DDR5_COMPONENT", "LPDDR5X"])
        result = apply_filter(sample_df, f)
        assert result.height == 4  # P1, P2, P3, P5

    def test_gte(self, sample_df):
        f = FilterExpression(field="data_rate_mt_s", op="gte", value=6400)
        result = apply_filter(sample_df, f)
        assert result.height == 4  # P2, P3, P4, P5

    def test_lte(self, sample_df):
        f = FilterExpression(field="data_rate_mt_s", op="lte", value=4800)
        result = apply_filter(sample_df, f)
        assert result.height == 1  # P1

    def test_between(self, sample_df):
        f = FilterExpression(field="data_rate_mt_s", op="between", low=6400, high=7200)
        result = apply_filter(sample_df, f)
        assert result.height == 3  # P2(6400), P4(6400), P5(7200)

    def test_contains_range(self, sample_df):
        # "supports 55C": min <= 55 AND max >= 55
        f = FilterExpression(
            field="operating_temp_min_c", op="contains_range", value=55,
            min_field="operating_temp_min_c", max_field="operating_temp_max_c",
        )
        result = apply_filter(sample_df, f)
        # All products have max >= 55, and all have min <= 55 (min is -40 or 0)
        assert result.height == 5

    def test_contains_range_excludes_out_of_range(self, sample_df):
        # "supports 100C": only products with max >= 100
        f = FilterExpression(
            field="operating_temp_min_c", op="contains_range", value=100,
            min_field="operating_temp_min_c", max_field="operating_temp_max_c",
        )
        result = apply_filter(sample_df, f)
        # Only P3 has max=105 >= 100
        assert result.height == 1
        assert result["product_id"][0] == "P3"

    def test_contains_range_boundary_inclusive(self, sample_df):
        # At the exact max boundary (95): products with max >= 95 should match
        f = FilterExpression(
            field="operating_temp_min_c", op="contains_range", value=95,
            min_field="operating_temp_min_c", max_field="operating_temp_max_c",
        )
        result = apply_filter(sample_df, f)
        # P1(95), P2(95), P4(95), P5(95) have max=95; P3 has max=105
        assert result.height == 5


# ---------------------------------------------------------------------------
# Filter combination tests
# ---------------------------------------------------------------------------

class TestFilterCombination:
    def test_and_combination(self, sample_df):
        filters = [
            FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
            FilterExpression(field="bus_width_bits", op="eq", value=8),
            FilterExpression(field="data_rate_mt_s", op="gte", value=6400),
        ]
        result = apply_filters(sample_df, filters)
        # P1(4800, excluded by gte), P2(6400, x8, DDR5 -> match), P5(7200, x16, excluded)
        assert result.height == 1
        assert result["product_id"][0] == "P2"

    def test_unknown_field_raises(self, sample_df):
        f = FilterExpression(field="nonexistent", op="eq", value=1)
        with pytest.raises(ValueError, match="unknown filter column"):
            apply_filters(sample_df, [f])


# ---------------------------------------------------------------------------
# Semantic filter expansion tests
# ---------------------------------------------------------------------------

class TestSemanticExpansion:
    def test_range_contains_expansion(self, sample_df, search_config):
        filters = [FilterExpression(field="operating_temperature", op="semantic", value=100)]
        expanded = expand_semantic(filters, search_config)
        assert len(expanded) == 1
        assert expanded[0].op == Operator.CONTAINS_RANGE
        assert expanded[0].min_field == "operating_temp_min_c"
        assert expanded[0].max_field == "operating_temp_max_c"
        assert expanded[0].value == 100
        # Apply the expanded filter
        result = apply_filters(sample_df, expanded)
        assert result.height == 1  # Only P3 (max=105)

    def test_expands_to_expansion(self, sample_df, search_config):
        filters = [FilterExpression(field="industrial_temperature", op="semantic")]
        expanded = expand_semantic(filters, search_config)
        assert len(expanded) == 2
        assert expanded[0].op == Operator.LTE
        assert expanded[0].field == "operating_temp_min_c"
        assert expanded[0].value == -40
        assert expanded[1].op == Operator.GTE
        assert expanded[1].field == "operating_temp_max_c"
        assert expanded[1].value == 95
        # Apply: products with min <= -40 AND max >= 95
        result = apply_filters(sample_df, expanded)
        # P1(-40,95), P3(-40,105), P5(-40,95) — P2(0,95) and P4(0,95) excluded (min=0 > -40)
        assert result.height == 3

    def test_unknown_semantic_raises(self, search_config):
        filters = [FilterExpression(field="nonexistent", op="semantic")]
        with pytest.raises(ValueError, match="unknown semantic filter"):
            expand_semantic(filters, search_config)

    def test_non_semantic_passes_through(self, sample_df, search_config):
        filters = [FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT")]
        expanded = expand_semantic(filters, search_config)
        assert len(expanded) == 1
        assert expanded[0].field == "product_family"


# ---------------------------------------------------------------------------
# from_simple_dict tests (sample_rows compatibility)
# ---------------------------------------------------------------------------

class TestFromSimpleDict:
    def test_scalar_becomes_eq(self):
        result = from_simple_dict({"col": "value"})
        assert len(result) == 1
        assert result[0].op == Operator.EQ
        assert result[0].value == "value"

    def test_list_becomes_in(self):
        result = from_simple_dict({"col": ["a", "b"]})
        assert len(result) == 1
        assert result[0].op == Operator.IN
        assert result[0].value == ["a", "b"]

    def test_empty_returns_empty(self):
        assert from_simple_dict(None) == []
        assert from_simple_dict({}) == []


# ---------------------------------------------------------------------------
# search_dataset service tests
# ---------------------------------------------------------------------------

class TestSearchDataset:
    def test_basic_search(self, temp_registry, search_config):
        filters = [
            FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
            FilterExpression(field="bus_width_bits", op="eq", value=8),
        ]
        result = search_dataset(temp_registry, "test_catalog", filters, limit=10,
                                 search_config=search_config)
        assert result["matched"] == 2  # P1, P2
        assert result["returned"] == 2
        assert "product_id" in result["columns"]

    def test_limit_cap(self, temp_registry, search_config):
        result = search_dataset(temp_registry, "test_catalog", [], limit=10000)
        assert result["returned"] <= 1000  # capped

    def test_sort(self, temp_registry, search_config):
        result = search_dataset(
            temp_registry, "test_catalog", [],
            sort=[SortExpression(field="data_rate_mt_s", descending=True)],
            limit=10,
        )
        rates = [r["data_rate_mt_s"] for r in result["rows"]]
        assert rates == sorted(rates, reverse=True)

    def test_columns_subset(self, temp_registry, search_config):
        result = search_dataset(
            temp_registry, "test_catalog", [],
            columns=["product_id", "data_rate_mt_s"], limit=5,
        )
        assert result["columns"] == ["product_id", "data_rate_mt_s"]

    def test_applied_filters_audit(self, temp_registry, search_config):
        filters = [FilterExpression(field="operating_temperature", op="semantic", value=100)]
        result = search_dataset(temp_registry, "test_catalog", filters, limit=5,
                                 search_config=search_config)
        assert "applied_filters" in result
        assert len(result["applied_filters"]) == 1
        assert result["applied_filters"][0]["op"] == "contains_range"

    def test_unknown_dataset_raises(self, temp_registry):
        with pytest.raises(ValueError, match="not registered"):
            search_dataset(temp_registry, "nonexistent", [])

    def test_unknown_sort_column_raises(self, temp_registry):
        with pytest.raises(ValueError, match="unknown sort column"):
            search_dataset(
                temp_registry, "test_catalog", [],
                sort=[SortExpression(field="nonexistent")], limit=5,
            )


# ---------------------------------------------------------------------------
# get_search_contract tests
# ---------------------------------------------------------------------------

class TestSearchContract:
    def test_contract_returns_fields(self, temp_registry, search_config):
        contract = get_search_contract(temp_registry, "test_catalog",
                                        search_config=search_config)
        assert contract["dataset_id"] == "test_catalog"
        assert "eq" in contract["supported_operators"]
        assert "semantic" in contract["supported_operators"]
        assert len(contract["searchable_fields"]) == 5
        assert len(contract["semantic_filters"]) == 2

    def test_contract_enriches_with_profile(self, temp_registry, search_config):
        profile = {
            "columns": {
                "product_family": {
                    "name": "product_family", "type": "categorical",
                    "top_values": [{"value": "DDR5_COMPONENT", "count": 3}],
                },
                "data_rate_mt_s": {
                    "name": "data_rate_mt_s", "type": "numeric",
                    "min": 4800, "max": 8533,
                },
            },
        }
        contract = get_search_contract(temp_registry, "test_catalog",
                                        search_config=search_config, profile=profile)
        # Find the product_family field
        pf = next(f for f in contract["searchable_fields"] if f["field"] == "product_family")
        assert pf["type"] == "categorical"
        assert "DDR5_COMPONENT" in pf["categorical_values"]
        # Find the data_rate field
        dr = next(f for f in contract["searchable_fields"] if f["field"] == "data_rate_mt_s")
        assert dr["type"] == "numeric"
        assert dr["range"]["min"] == 4800
        assert dr["range"]["max"] == 8533

    def test_contract_unknown_dataset_raises(self, temp_registry, search_config):
        with pytest.raises(ValueError, match="not registered"):
            get_search_contract(temp_registry, "nonexistent", search_config=search_config)
