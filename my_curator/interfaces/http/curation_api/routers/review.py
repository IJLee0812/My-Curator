"""GET /v1/review and PATCH /v1/clips/{id}/review — P3-5 Verify-by-Exception."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from my_curator.adapters.storage.pg import PGRepository

from ..deps import get_pg

router = APIRouter()

_VALID_ACTIONS = {"approve", "reject", "pending"}
_ACTION_TO_STATE = {"approve": "approved", "reject": "rejected", "pending": "pending"}


class ReviewAction(BaseModel):
    action: str  # "approve" | "reject"


class ReviewQueueItem(BaseModel):
    queue_id: int
    clip_id: str
    state: str
    reviewed_at: str | None
    reason: str | None
    created_at: str
    blob_uri: str
    frames_blob_uri: str | None
    start_s: float
    end_s: float
    dna_json: dict[str, Any] | None


class ReviewQueueResponse(BaseModel):
    items: list[ReviewQueueItem]
    total: int


@router.get("/v1/review", response_model=ReviewQueueResponse)
async def list_review_queue(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    pg: PGRepository = Depends(get_pg),
) -> ReviewQueueResponse:
    rows = await pg.get_review_queue(status=status, limit=limit)
    items = [
        ReviewQueueItem(
            queue_id=r["queue_id"],
            clip_id=str(r["clip_id"]),
            state=r["state"],
            reviewed_at=r["reviewed_at"].isoformat() if r["reviewed_at"] else None,
            reason=r.get("reason"),
            created_at=r["created_at"].isoformat(),
            blob_uri=r["blob_uri"],
            frames_blob_uri=r.get("frames_blob_uri"),
            start_s=r["start_s"],
            end_s=r["end_s"],
            dna_json=r.get("dna_json"),
        )
        for r in rows
    ]
    return ReviewQueueResponse(items=items, total=len(items))


@router.patch("/v1/clips/{clip_id}/review")
async def review_clip(
    clip_id: str,
    body: ReviewAction,
    pg: PGRepository = Depends(get_pg),
) -> dict[str, str]:
    if body.action not in _VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"action must be one of {sorted(_VALID_ACTIONS)}",
        )
    try:
        uid = UUID(clip_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="clip_id must be a valid UUID") from None

    state = _ACTION_TO_STATE[body.action]
    try:
        await pg.set_review_status(uid, state)
    except asyncpg.ForeignKeyViolationError:
        raise HTTPException(status_code=404, detail="clip not found") from None

    return {"clip_id": clip_id, "state": state}
