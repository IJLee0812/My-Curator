from __future__ import annotations

from my_curator.domain.scout.base import ScoutReport


class BestOfNAggregator:
    """Selects the best-grounded ScoutReport from N candidates.

    Scoring: count of YOLO inventory class names that appear (as substring)
    in the report text. Tie-breaking: lowest temperature (most deterministic) wins.
    Validated paradigm: Semantic-Drive arXiv:2512.12012 §3.2.
    """

    def score(self, report: ScoutReport, inventory: dict[str, int]) -> int:
        """Return the YOLO class-presence overlap score for a single report.

        Both inventory keys and report text are lowercased before comparison
        so caller-side capitalisation does not affect the result.
        """
        text = report.text.lower()
        return sum(1 for cls in inventory if cls.lower() in text)

    def select(
        self,
        reports: list[ScoutReport],
        inventory: dict[str, int],
    ) -> ScoutReport | None:
        """Return the report with highest YOLO class-presence overlap.

        Returns None if reports is empty. Returns the sole report directly
        if len(reports) == 1. On tie or zero score, lowest temperature wins.
        """
        if not reports:
            return None
        if len(reports) == 1:
            return reports[0]

        return min(reports, key=lambda r: (-self.score(r, inventory), r.temperature))
