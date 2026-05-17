"""Shim — EmbedderWorker moved to my_curator.application.workers.embedder_worker;
CLI entrypoint moved to my_curator.cli.run_embedder.  Removed in R-7.
"""

from my_curator.application.workers.embedder_worker import EmbedderWorker  # noqa: F401
from my_curator.cli.run_embedder import (  # noqa: F401
    _build_arg_parser,
    _run,
    main,
)

__all__ = ["EmbedderWorker", "main"]


if __name__ == "__main__":
    main()
