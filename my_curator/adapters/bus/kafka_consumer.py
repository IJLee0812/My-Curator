"""Reusable KafkaConsumer loop wrapper.

A thin layer around ``kafka.KafkaConsumer`` so application-level consumers
(``CurationConsumer``, ``EmbedderWorker``) only have to provide a
``(topic, value) -> Awaitable[None]`` callback.  Behavior preserved from the
original loops in ``src/bus/kafka.py:_run`` and
``services/embedder/worker.py:_run``: per-message try/except + log.exception,
``consumer_timeout_ms`` semantics (``0`` / ``float('inf')`` = run forever),
graceful KeyboardInterrupt drain.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


async def run_consumer_loop(
    handler: MessageHandler,
    *,
    topics: list[str] | tuple[str, ...],
    broker: str,
    group_id: str,
    timeout_ms: int | float = 300000,
    auto_offset_reset: str = "earliest",
) -> None:
    """Subscribe to *topics* and dispatch each message to *handler(topic, value)*.

    Args:
        handler: async callback that receives (topic, value_dict).
        topics: Kafka topic names to subscribe to.
        broker: bootstrap servers string.
        group_id: consumer group id.
        timeout_ms: idle timeout in ms; ``0`` or ``float('inf')`` means run forever.
        auto_offset_reset: ``"earliest"`` (default) or ``"latest"``.
    """
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("kafka-python not installed") from exc

    if timeout_ms == 0:
        _timeout: float | int = float("inf")
    else:
        _timeout = timeout_ms

    kafka = KafkaConsumer(
        *topics,
        bootstrap_servers=broker,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        group_id=group_id,
        consumer_timeout_ms=_timeout,
    )
    log.info("Listening on %s @ %s (group=%s)", list(topics), broker, group_id)

    try:
        for message in kafka:
            try:
                await handler(message.topic, message.value)
            except Exception:
                log.exception("Unhandled error on message offset %d", message.offset)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        kafka.close()
