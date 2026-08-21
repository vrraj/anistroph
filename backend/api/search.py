"""Search API routes — parametric search, search contract, and predict-on-search."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import PredictOnSearchRequest, SearchRequest
from backend.search.filters import FilterExpression, SortExpression
from backend.services import get_services

router = APIRouter(prefix="/datasets", tags=["search"])


@router.get("/{dataset_id}/search-contract")
async def get_search_contract(dataset_id: str):
    """Return the self-describing search contract for a dataset.

    Lists searchable fields (types, units, operators, aliases, categorical
    values / numeric ranges) and semantic filters. Use this before calling
    /datasets/{dataset_id}/search to discover what filters are available.
    """
    svc = get_services()
    try:
        return svc.get_search_contract(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/search")
async def search(dataset_id: str, req: SearchRequest):
    """Run a deterministic structured search over a dataset.

    Supports operators eq, in, gte, lte, between, contains_range, plus
    semantic filter expansion. Returns matching rows with an audit of the
    applied (normalized) filters.
    """
    svc = get_services()
    try:
        filters = [FilterExpression(**f.model_dump()) for f in req.filters]
        sort = [SortExpression(**s.model_dump()) for s in req.sort] if req.sort else None
        return svc.search(
            dataset_id, filters, sort=sort, limit=req.limit, columns=req.columns,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/predict-on-search")
async def predict_on_search(dataset_id: str, req: PredictOnSearchRequest):
    """Search a catalog dataset, then predict for each matching product.

    Runs a parametric search on the catalog dataset, then for each matching
    product_id invokes the specified model using entity-lookup prediction.
    Results are enriched with the prediction and ranked by prediction outcome
    (descending: highest risk probability or longest lead time first).
    """
    svc = get_services()
    try:
        filters = [FilterExpression(**f.model_dump()) for f in req.filters]
        sort = [SortExpression(**s.model_dump()) for s in req.sort] if req.sort else None
        return svc.predict_on_search(
            search_dataset_id=dataset_id,
            model_id=req.model_id,
            filters=filters,
            sort=sort,
            limit=req.limit,
            columns=req.columns,
            timestamp=req.timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
