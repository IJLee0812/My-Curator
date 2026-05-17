"""GT-9: golden frame upload pool shutdown drain.

Verifies that ``VLMKafkaSignalPublisher.close()`` drains in-flight uploads:
``self._upload_executor.shutdown(wait=True)`` must wait for submitted tasks to
finish before returning, so no upload is silently dropped on pipeline exit.

References:
  docs/refactoring_plan.md  §3.1 GT-9.
"""

from __future__ import annotations

import threading
import time

import pytest


def _make_publisher():
    from vllm_ds_app_kafka_publish import VLMKafkaSignalPublisher

    return VLMKafkaSignalPublisher({}, "topic", dry_run=True)


@pytest.mark.unit
class TestGoldenUploadDrain:
    def test_executor_exists_after_init(self):
        pub = _make_publisher()
        assert pub._upload_executor is not None
        pub.close()

    def test_shutdown_waits_for_in_flight_uploads(self):
        """All submitted uploads must complete before close() returns."""
        pub = _make_publisher()

        completed = []
        slow_task_done = threading.Event()

        def slow_upload():
            time.sleep(0.15)
            completed.append("slow")
            slow_task_done.set()

        def fast_upload():
            completed.append("fast")

        pub._upload_executor.submit(slow_upload)
        pub._upload_executor.submit(fast_upload)

        pub.close()  # must drain — wait=True

        assert slow_task_done.is_set(), "slow upload did not complete before close()"
        assert "slow" in completed
        assert "fast" in completed
        assert len(completed) == 2

    def test_subsequent_submit_after_close_raises(self):
        """After close()'s shutdown, the executor refuses new work — caller protection."""
        pub = _make_publisher()
        pub.close()
        with pytest.raises(RuntimeError):
            pub._upload_executor.submit(lambda: None)

    def test_multiple_close_calls_safe(self):
        """Idempotent close — second invocation must not raise."""
        pub = _make_publisher()
        pub.close()
        pub.close()  # no exception
