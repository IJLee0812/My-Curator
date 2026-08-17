"""P4-7 3-way (video / text / hybrid) retrieval benchmark harness.

Ground truth per query is a structured DNA predicate (see retrieval_bench.json),
resolved against the live corpus at run time — corpus-independent and not
gameable by a single clip.  ``recall@5 = |top5 ∩ relevant| / min(5, |relevant|)``,
averaged over queries; gate: the best modality's average ≥ 0.85.

The predicate evaluator is pure and unit-tested here on the host.  The bench
itself needs torch (text tower) + a populated hybrid Milvus collection, so it is
skipped unless the stack is up and re-embed has run.  Run in-container::

    pytest tests/performance/test_retrieval_bench.py -m performance -v -s
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

_BENCH_PATH = Path(__file__).parent / "retrieval_bench.json"
_MILVUS_URI = "http://localhost:19530"
_K = 5


# ── pure predicate evaluator (host-testable) ────────────────────────────────────


def _leaf_match(dna: dict, cond: dict) -> bool:
    val: Any = dna
    for key in cond["path"]:
        if not isinstance(val, dict):
            return False
        val = val.get(key)
    if "in" in cond:
        return val in cond["in"]
    if "ilike_any" in cond:
        return isinstance(val, str) and any(s.lower() in val.lower() for s in cond["ilike_any"])
    return False


def predicate_matches(dna: dict, node: dict) -> bool:
    """Evaluate an all/any/leaf relevance predicate against one DNA dict."""
    if "all" in node:
        return all(predicate_matches(dna, c) for c in node["all"])
    if "any" in node:
        return any(predicate_matches(dna, c) for c in node["any"])
    return _leaf_match(dna, node)


def recall_at_k(top: list, relevant: set, k: int = _K) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in top[:k] if cid in relevant)
    return hits / min(k, len(relevant))


# ── evaluator unit coverage (runs on host, no deps) ─────────────────────────────


def test_predicate_all_any_leaf():
    dna = {
        "topology": {"road_type": "motorway"},
        "planner_logic": {
            "ego_maneuver": "cruise",
            "safety_event": {"event_type": "near_miss"},
            "risk_level": "elevated",
        },
        "odd": {"lighting": "day"},
        "scene_description": "A cyclist rides beside the ego vehicle.",
    }
    assert predicate_matches(dna, {"path": ["topology", "road_type"], "in": ["motorway"]})
    assert not predicate_matches(dna, {"path": ["topology", "road_type"], "in": ["parking"]})
    assert predicate_matches(dna, {"path": ["scene_description"], "ilike_any": ["CYCLIST"]})
    assert predicate_matches(
        dna,
        {
            "any": [
                {"path": ["planner_logic", "ego_maneuver"], "in": ["brake_hard"]},
                {"path": ["planner_logic", "safety_event", "event_type"], "in": ["near_miss"]},
            ]
        },
    )
    assert not predicate_matches(dna, {"path": ["missing", "key"], "in": ["x"]})


def test_recall_at_k_math():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=5) == pytest.approx(0.5)  # 1 / min(5,2)
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0
    assert (
        recall_at_k(["a", "b", "c", "d", "e", "f"], {"a", "b", "c", "d", "e", "f", "g"}, k=5) == 1.0
    )


def test_bench_spec_is_wellformed():
    spec = json.loads(_BENCH_PATH.read_text())
    assert spec["k"] == _K
    ids = [q["id"] for q in spec["queries"]]
    assert len(ids) == len(set(ids)) == 5
    for q in spec["queries"]:
        assert q["query"] and q["tier"] and "relevant" in q


# ── the 3-way benchmark (stack + torch required; skips otherwise) ────────────────


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.performance
@pytest.mark.asyncio
async def test_three_way_recall_bench():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available — run the bench in the curation-api/embedder container")
    if not _tcp_open("127.0.0.1", 19530):
        pytest.skip("Milvus not up")

    from my_curator.adapters.embed.text_tower import CosmosEmbed1Encoder
    from my_curator.adapters.storage.milvus import MilvusHybridRepository
    from my_curator.adapters.storage.pg import PGRepository, dsn_from_env

    hybrid = await MilvusHybridRepository.create(_MILVUS_URI)
    if await hybrid.count() == 0:
        await hybrid.close()
        pytest.skip("hybrid collection empty — run `python -m my_curator.cli.reembed_corpus` first")

    spec = json.loads(_BENCH_PATH.read_text())
    pg = await PGRepository.create(dsn_from_env())
    rows = await pg.list_dna(limit=1000)
    encoder = CosmosEmbed1Encoder()

    modalities = ("text", "video", "hybrid")
    sums = {m: 0.0 for m in modalities}
    n_queries = 0

    try:
        for q in spec["queries"]:
            relevant = {
                r["clip_id"] for r in rows if predicate_matches(r["dna_json"], q["relevant"])
            }
            if not relevant:
                print(f"[bench] {q['id']}: no relevant clips in corpus — skipped")
                continue
            n_queries += 1
            qv = encoder.encode_text(q["query"])
            per = {}
            for m in modalities:
                if m == "text":
                    hits = await hybrid.search_text(qv, top_k=_K)
                elif m == "video":
                    hits = await hybrid.search_video(qv, top_k=_K)
                else:
                    hits = await hybrid.hybrid_search(text_vec=qv, video_vec=qv, top_k=_K)
                top = [h["clip_id"] for h in hits]
                r = recall_at_k(top, relevant, _K)
                per[m] = r
                sums[m] += r
            print(
                f"[bench] {q['id']:<28} |relevant|={len(relevant):>3} "
                + " ".join(f"{m}={per[m]:.2f}" for m in modalities)
            )
    finally:
        await pg.close()
        await hybrid.close()

    assert n_queries > 0, "no queries had relevant clips — check corpus / predicates"
    avgs = {m: sums[m] / n_queries for m in modalities}
    winner = max(avgs, key=avgs.get)
    print("\n[bench] avg recall@5:  " + "  ".join(f"{m}={avgs[m]:.3f}" for m in modalities))
    print(f"[bench] winner = {winner} ({avgs[winner]:.3f}); gate avg >= 0.85")

    assert avgs[winner] >= 0.85, (
        f"best modality {winner} avg recall@5 = {avgs[winner]:.3f} < 0.85 "
        f"(all: {', '.join(f'{m}={avgs[m]:.3f}' for m in modalities)})"
    )
