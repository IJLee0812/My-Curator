"""Recall@5 benchmark — P3-3.

Loads the 14-clip gold set and verifies that for each clip a natural-language
query derived from its DNA labels returns that clip in the top-5 Milvus ANN
results served by the curation-api at /v1/search.

Pass criterion: Recall@5 >= 0.80  (at least 12 / 14 clips retrieved).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"
API_BASE = "http://localhost:8001"
RECALL_K = 5
RECALL_THRESHOLD = 0.80

_WEATHER_TEXT: dict[str, str] = {
    "clear": "clear",
    "rain": "rainy",
    "light_rain": "light rain",
    "snow": "snowy",
    "heavy_snow": "heavy snowfall",
}
_LIGHTING_TEXT: dict[str, str] = {
    "day": "daytime",
    "night": "nighttime",
    "dusk": "dusk",
    "tunnel": "tunnel interior",
}
_ROAD_TEXT: dict[str, str] = {
    "urban": "urban road",
    "primary": "primary road",
    "motorway": "motorway highway",
    "residential": "residential street",
    "parking": "parking lot",
}


def _build_query(clip: dict) -> str:
    w = _WEATHER_TEXT.get(clip["weather"], clip["weather"])
    li = _LIGHTING_TEXT.get(clip["lighting"], clip["lighting"])
    r = _ROAD_TEXT.get(clip["road_type"], clip["road_type"])
    return f"{w} {li} {r} driving scene"


@pytest.mark.performance
async def test_recall_at_5() -> None:
    gold = json.loads(GOLD_SET_PATH.read_text())
    clips = gold["clips"]

    hits = 0
    misses: list[dict] = []

    async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client:
        for clip in clips:
            query = _build_query(clip)
            resp = await client.post(
                "/v1/search",
                json={
                    "query": query,
                    "filters": {"weather": clip["weather"], "lighting": clip["lighting"]},
                    "limit": RECALL_K,
                    "top_k": 1000,
                },
            )
            resp.raise_for_status()
            result_ids = [r["clip_id"] for r in resp.json()["results"]]

            if clip["clip_id"] in result_ids:
                hits += 1
            else:
                misses.append(
                    {
                        "clip_id": clip["clip_id"],
                        "video": clip["video"],
                        "query": query,
                        f"top{RECALL_K}": result_ids,
                    }
                )

    recall = hits / len(clips)

    if misses:
        print(f"\nRecall@{RECALL_K} misses ({len(misses)} / {len(clips)}):")
        for m in misses:
            print(f"  clip_id : {m['clip_id']}")
            print(f"  video   : {m['video']}")
            print(f"  query   : {m['query']}")
            print(f"  top{RECALL_K}    : {m[f'top{RECALL_K}']}")

    print(
        f"\nRecall@{RECALL_K} = {hits}/{len(clips)} = {recall:.3f}  (threshold={RECALL_THRESHOLD})"
    )

    assert recall >= RECALL_THRESHOLD, (
        f"Recall@{RECALL_K} = {recall:.3f} < {RECALL_THRESHOLD} "
        f"({hits}/{len(clips)} hits; {len(misses)} misses)"
    )
