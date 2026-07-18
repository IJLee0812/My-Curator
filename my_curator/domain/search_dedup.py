"""Adjacent-window / source de-duplication for search results (P4-7).

Pure domain logic (stdlib only) over ranked result dicts.  Two modes:

- ``dedup_adjacent_windows`` (always-on, conservative): P4-5 gives each source
  clip several overlapping ~5 s windows, so a query can match near-identical
  windows of the same footage and fill several top-k slots.  Within one
  ``source_clip_id`` this drops a lower-ranked window whose query-similarity
  score is within ``score_epsilon`` of an already-kept window from the same
  source — collapsing near-duplicate windows while keeping windows that score
  differently (i.e. distinct events / content) for that source.

  This uses same-source + near-equal query similarity as a stand-in for the
  "embedding cosine ≥ threshold" near-duplicate test: near-identical windows of
  one source produce near-identical similarity to any query, and needs no extra
  Milvus round-trip.  ``score_epsilon`` is the tunable threshold (empirically
  set on the real corpus — P4-7 E2).

- ``dedup_by_source`` (opt-in, ``dedup_by_source`` query param): collapse every
  ``source_clip_id`` to its single top-ranked hit — maximum result diversity.

Both assume ``results`` is already sorted by descending score and preserve that
order.  Items without a ``source_clip_id`` (or, for adjacent-window, without a
score) are always kept.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_SCORE_EPSILON = 1e-3


def dedup_adjacent_windows(
    results: list[dict[str, Any]], *, score_epsilon: float = _DEFAULT_SCORE_EPSILON
) -> list[dict[str, Any]]:
    """Drop same-source windows whose score ≈ a kept window's (near-duplicates)."""
    kept: list[dict[str, Any]] = []
    scores_by_source: dict[str, list[float]] = {}
    for r in results:
        source = r.get("source_clip_id")
        score = r.get("score")
        if source is None or score is None:
            kept.append(r)
            continue
        score = float(score)
        prior = scores_by_source.get(source, [])
        if any(abs(score - ks) <= score_epsilon for ks in prior):
            continue
        kept.append(r)
        scores_by_source.setdefault(source, []).append(score)
    return kept


def dedup_by_source(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the top-ranked hit per ``source_clip_id`` (result diversity)."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for r in results:
        source = r.get("source_clip_id")
        if source is None:
            kept.append(r)
            continue
        if source in seen:
            continue
        seen.add(source)
        kept.append(r)
    return kept
