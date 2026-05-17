###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
###################################################################################################

"""Shim — CurationConsumer moved to my_curator.application.consumers.curation_consumer;
KafkaConsumer loop moved to my_curator.adapters.bus.kafka_consumer;
CLI argparse + _run moved to my_curator.cli.run_curation_consumer.  Removed in R-7.

A local ``main()`` is preserved here so existing monkeypatch points
(``_SCOUT_PROMPT_PATH``, ``_compute_prompt_hash``, ``_run``) on this shim
module keep working.  Symbols are re-exported from the new locations.
"""

import asyncio
import logging
import sys

from my_curator.application.consumers.curation_consumer import (  # noqa: F401
    _SCOUT_PROMPT_PATH,
    PIPELINE_VERSION,
    CurationConsumer,
    _compute_prompt_hash,
    _parse_dna_json,
)
from my_curator.cli.run_curation_consumer import (  # noqa: F401
    _build_arg_parser,
    _run,
)
from my_curator.domain.scout.versioning import assert_prompt_registered

log = logging.getLogger(__name__)

__all__ = [
    "PIPELINE_VERSION",
    "CurationConsumer",
    "main",
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_arg_parser().parse_args()

    if not _SCOUT_PROMPT_PATH.exists():
        log.error("Scout prompt file not found: %s", _SCOUT_PROMPT_PATH)
        sys.exit(1)

    scout_prompt_hash = _compute_prompt_hash(_SCOUT_PROMPT_PATH)
    log.info("Scout prompt hash: %s (from %s)", scout_prompt_hash, _SCOUT_PROMPT_PATH.name)

    try:
        assert_prompt_registered(scout_prompt_hash)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    asyncio.run(_run(args, scout_prompt_hash))


if __name__ == "__main__":
    main()
