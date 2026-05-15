"""Video streaming helpers for curation-api (P3-4+).

Public surface:
  base.py        — resolve file:// blob_uri → FileResponse (Accept-Ranges)
  timestamp.py   — .timestamp sidecar parsing for frame-accurate seeking
  minio.py       — minio:// / legacy blob_uri → presigned URL
"""
