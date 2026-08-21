"""Search API routes — parametric search and search contract."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import SearchRequest
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
