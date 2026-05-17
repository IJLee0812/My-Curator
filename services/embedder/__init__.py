"""Backwards-compatibility shim — moved to my_curator.application.workers
(EmbedderWorker) + my_curator.adapters.embed.video_tower (CosmosEmbed1)
+ my_curator.adapters.storage.frame_loader (load_frames).  Removed in R-7.
"""

import warnings

warnings.warn(
    "services.embedder is moving to my_curator.application.workers + "
    "my_curator.adapters.{embed,storage}; update imports.",
    DeprecationWarning,
    stacklevel=2,
)
