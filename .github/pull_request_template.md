<!--
PR Title convention: [PX-Y] Short imperative summary
  e.g. [P2-3] Best-of-N aggregator with YOLO26 symbolic reward
Reviewer is auto-requested via CODEOWNERS; assignee is auto-set by the
auto-assign workflow. Do not remove the sections below.
-->

## Summary
<!-- One or two sentences. What and why. -->

## Linked issue
Closes #

## Phase / PR slot
<!-- e.g. "Phase 2 · P2-3". -->

## Changes
- 
- 

## Test plan
- [ ] `pytest -m "unit"` green
- [ ] `pytest -m "integration"` green (if applicable)
- [ ] New tests added under the correct marker (`unit` / `integration` / `e2e` / `schema` / `performance` / `simulation` / `prompt_regression`)
- [ ] `ruff check . && ruff format --check .` clean
- [ ] Existing 238 baseline tests still green (plus 129 schema from P1-1 onward)

## Definition-of-Done checklist
- [ ] All baseline + new-tier tests green
- [ ] Lint clean
- [ ] If prompt/schema touched → `prompt_regression` + `schema` markers green; `dna_version` bumped if needed
- [ ] Docs updated (README / runbook) when user-visible
- [ ] GPU0 VRAM not inflated ≥ 1 GB w/o justification
- [ ] No new SSD ≥ 5 GB w/o lifecycle policy
- [ ] Reviewer approval received (auto-requested by CODEOWNERS)

## Notes for reviewer
<!-- Architecture decisions, tradeoffs, follow-ups filed as issues. -->
