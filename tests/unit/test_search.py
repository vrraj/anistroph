"""Unit tests for the generic parametric search engine."""

from __future__ import annotations

import pytest
import polars as pl

from backend.search.filters import (
    FilterExpression, Operator, SortExpression,
    apply_filter, apply_filters, expand_semantic, from_simple_dict,
)
from backend.search.spec import SearchConfig, SearchFieldSpec, SemanticFilterSpec
from backend.search.service import get_search_contract, search_dataset
from backend.datasets.registry import DatasetMeta, DatasetRegistry


@pytest.fixture
def sample_df() -> pl.DataFrame:
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
            "product_family": SearchFieldSpec(field="product_family", operators=["eq", "in"]),
            "data_rate_mt_s": SearchFieldSpec(field="data_rate_mt_s", operators=["eq", "in", "gte", "lte", "between"], unit="MT/s"),
            "bus_width_bits": SearchFieldSpec(field="bus_width_bits", operators=["eq", "in", "gte", "lte"]),
            "component_density_gb": SearchFieldSpec(field="component_density_gb", operators=["eq", "gte", "lte", "between"], unit="Gb"),
            "part_status": SearchFieldSpec(field="part_status", operators=["eq", "in"]),
        },
        semantic_filters={
            "operating_temperature": SemanticFilterSpec(
                name="operating_temperature", type="range_contains",
                min_field="operating_temp_min_c", max_field="operating_temp_max_c", unit="C",
            ),
            "industrial_temperature": SemanticFilterSpec(
                name="industrial_temperature", type="expands_to",
                expands_to=[
                    {"field": "operating_temp_min_c", "op": "lte", "value": -40},
                    {"field": "operating_temp_max_c", "op": "gte", "value": 95},
                ],
            ),
        },
    )


@pytest.fixture
def temp_registry(tmp_path, sample_df) -> DatasetRegistry:
    pq_path = tmp_path / "test_catalog.parquet"
    sample_df.write_parquet(str(pq_path))
    reg = DatasetRegistry(tmp_path / "dataset_registry.json")
    reg._datasets["test_catalog"] = DatasetMeta(
        dataset_id="test_catalog", name="Test", source="test",
        row_count=sample_df.height, columns=list(sample_df.columns),
        entity_key="product_id", parquet_path=str(pq_path),
    )
    reg._save()
    return reg


class TestOperators:
    def test_eq_and_in(self, sample_df):
        assert apply_filter(sample_df, FilterExpression(field="product_family", op="eq", value="LPDDR5X")).height == 1
        assert apply_filter(sample_df, FilterExpression(field="product_family", op="in", value=["DDR5_COMPONENT", "LPDDR5X"])).height == 4

    def test_gte_lte_between(self, sample_df):
        assert apply_filter(sample_df, FilterExpression(field="data_rate_mt_s", op="gte", value=6400)).height == 4
        assert apply_filter(sample_df, FilterExpression(field="data_rate_mt_s", op="lte", value=4800)).height == 1
        assert apply_filter(sample_df, FilterExpression(field="data_rate_mt_s", op="between", low=6400, high=7200)).height == 3

    def test_contains_range(self, sample_df):
        f = FilterExpression(field="operating_temp_min_c", op="contains_range", value=100,
                             min_field="operating_temp_min_c", max_field="operating_temp_max_c")
        result = apply_filter(sample_df, f)
        assert result.height == 1 and result["product_id"][0] == "P3"

    def test_and_combination(self, sample_df):
        result = apply_filters(sample_df, [
            FilterExpression(field="product_family", op="eq", value="DDR5_COMPONENT"),
            FilterExpression(field="bus_width_bits", op="eq", value=8),
            FilterExpression(field="data_rate_mt_s", op="gte", value=6400),
        ])
        assert result.height == 1 and result["product_id"][0] == "P2"


class TestSemanticExpansion:
    def test_range_contains_expansion(self, sample_df, search_config):
        expanded = expand_semantic([FilterExpression(field="operating_temperature", op="semantic", value=100)], search_config)
        assert len(expanded) == 1 and expanded[0].op == Operator.CONTAINS_RANGE
        assert apply_filters(sample_df, expanded).height == 1

    def test_expands_to_expansion(self, sample_df, search_config):
        expanded = expand_semantic([FilterExpression(field="industrial_temperature", op="semantic")], search_config)
        assert len(expanded) == 2
        assert apply_filters(sample_df, expanded).height == 3  # P1, P3, P5

    def test_unknown_semantic_raises(self, search_config):
        with pytest.raises(ValueError, match="unknown semantic filter"):
            expand_semantic([FilterExpression(field="nonexistent", op="semantic")], search_config)


class TestSearchDataset:
    def test_basic_search_with_semantic(self, temp_registry, search_config):
        result = search_dataset(temp_registry, "test_catalog",
                                [FilterExpression(field="operating_temperature", op="semantic", value=100)],
                                limit=5, search_config=search_config)
        assert result["matched"] == 1
        assert result["applied_filters"][0]["op"] == "contains_range"

    def test_sort_and_columns(self, temp_registry, search_config):
        result = search_dataset(temp_registry, "test_catalog", [],
                                sort=[SortExpression(field="data_rate_mt_s", descending=True)],
                                columns=["product_id", "data_rate_mt_s"], limit=5)
        rates = [r["data_rate_mt_s"] for r in result["rows"]]
        assert rates == sorted(rates, reverse=True)
        assert result["columns"] == ["product_id", "data_rate_mt_s"]


class TestSearchContract:
    def test_contract_returns_fields(self, temp_registry, search_config):
        contract = get_search_contract(temp_registry, "test_catalog", search_config=search_config)
        assert len(contract["searchable_fields"]) == 5
        assert len(contract["semantic_filters"]) == 2
        assert "semantic" in contract["supported_operators"]
