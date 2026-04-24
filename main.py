#!/usr/bin/env python3
"""DeepStream-VLM Pipeline Entrypoint

Handles config initialization before plugin import to ensure -c/--config
is applied before gstnvvllmvlm.py's module-level get_config() call.

Modes:
  Pure VLM:           python3 main.py video.mp4 -c config.yaml
  VLM + Detection:    python3 main.py video.mp4 -c config.yaml --detect
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "plugin"))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Pre-parse -c/--config before any imports that trigger get_config()
config_path = None
for i, arg in enumerate(sys.argv):
    if arg in ("-c", "--config") and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
        break

if config_path:
    from config_loader import get_config

    get_config(config_path)

from vllm_ds_app_kafka_publish import main

if __name__ == "__main__":
    main()
