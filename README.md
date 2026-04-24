# My-Curator

> An **Autonomous Driving front-camera data curation & validation platform**, built on top of [DeepStream-VLM](https://github.com/IJLee0812/DeepStream-VLM) (NVIDIA DeepStream 9.0 + `nvvllmvlm` + Cosmos-Reason2-8B FP8 + **YOLO26** closed-vocab detector). Upstream YOLOE code paths are inherited but not used in My-Curator.

> **Status**: baseline pipeline is live (DeepStream-VLM, ported as-is). Expansion to Scenario-DNA schema + Scout + YOLO26 grounding + storage tri-stack (MinIO + Postgres + Milvus) + CARLA + MLOps is in progress (Judge deferred to post-v0.1).

---

## What it does (today)

A GStreamer pipeline that splits a driving video into temporal windows (e.g. 5-second segments) and, for each segment, produces a **schema-validated scene description** from a VLM. Three detector modes are available via `--detect-config`:

| Mode | `nvinfer` config | Vocabulary | Mask |
|---|---|---|---|
| Pure VLM | — | — | — |
| YOLO26 | `config_infer_yolo26.txt` | Closed (80 COCO) | No |
| YOLOE-26 detect | `config_infer_yolo26e.txt` | **Open-vocab** | No |
| YOLOE-26 seg | `config_infer_yolo26e_seg.txt` | **Open-vocab** | **Yes** (OSD) |

Pipeline topology (current):
```
uridecodebin × N → nvstreammux (batch=N) → [nvinfer(YOLO)]
                                         → nvvideoconvert
                                         → nvvllmvlm (Cosmos-Reason2-8B FP8)
                                         → Kafka + JSON (+ optional OSD MP4)
```

Multi-source (RTSP + file) is already supported via `uridecodebin` per stream and a per-`stream_id` `StreamContext` behind a single shared VLM instance.

---

## What it will do (expansion)

The baseline engine covers only the **captioner** step of an AD curation pipeline. My-Curator extends it into a full curation/validation system:

| Direction | Adds |
|---|---|
| **Up** (semantics) | **Scenario DNA v0.1** — queryable 4-layer ontology (ODD / Topology / Actor Dynamics / Planner Logic); schema in `schemas/scenario_dna_v0_1.schema.json` |
| **Sideways** (quality) | Neuro-symbolic **Scout + YOLO26 grounding** — single Cosmos-Reason2 Scout with N=3 temperature sampling + YOLO26 symbolic inventory + best-of-N aggregator (Judge LLM deferred to post-v0.1) |
| **Down** (infra) | MinIO + PostgreSQL (JSONB + GIN) + Milvus (GPU_CAGRA) + Kafka event bus |
| **Forward** (sim) | CARLA 0.9.15 integration + OpenSCENARIO corner-case library + GT-vs-Judge accuracy suite |
| **Around** (ops) | Prometheus + Grafana SLOs, tiered GH Actions CI, gold set indexed in `clips` table (no DVC), k3s/Helm deployment |

---

## Sample output

Left: middle frame of a 5-second segment. Right: VLM scene summary + YOLO-grounded key objects.

<p>
  <img src="assets/images/desc1.jpg" width="100%"><br>
  <img src="assets/images/desc2.jpg" width="100%"><br>
  <img src="assets/images/desc3.jpg" width="100%">
</p>

---

## Quickstart (baseline pipeline)

Prerequisites: Docker w/ NVIDIA Container Toolkit, NVIDIA driver 580+, an RTX 4090-class GPU (≥24 GB VRAM for the Cosmos-Reason2 FP8 checkpoint; target reference is **2× RTX 4090** where GPU 1 will host the expansion infra), NGC API key.

### 1. Clone and bring up the baseline stack
```bash
git clone https://github.com/IJLee0812/My-Curator.git
cd My-Curator
cp .env.example .env          # fill in NGC_API_KEY / HF_TOKEN
docker compose up -d          # ds9-vlm-dev + Kafka + Zookeeper
docker exec -it ds9-vlm-dev bash
```

### 2. Download the VLM (once, ~9 GB FP8)
```bash
# inside the container
ngc config set                 # paste NGC_API_KEY, org=nvidia
ngc registry model download-version \
    "nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8" \
    --dest /workspace/models/hub
```

### 3. Pick a detector and prepare its ONNX
```bash
# Option A — closed-vocab YOLO26 (80 COCO classes)
python3 scripts/download_model.py --model yolo26 --size m
python3 scripts/export_yolo26.py -w models/yolo26m.pt --simplify

# Option B — open-vocab YOLOE-26 detect
python3 scripts/download_model.py --model yoloe --size m
python3 scripts/export_yoloe.py \
    -w models/yoloe-26m-seg.pt \
    --custom-classes "vehicle,person,motorcycle,traffic_sign,traffic_light,truck,bus,bicycle" \
    --dynamic --simplify

# Option C — open-vocab YOLOE-26 seg (masks in OSD)
python3 scripts/export_yoloe_seg.py \
    -w models/yoloe-26m-seg.pt \
    --custom-classes "vehicle,person,motorcycle,traffic_sign,traffic_light,truck,bus,bicycle" \
    --dynamic --simplify --build-engine
```

> **Seg needs `--build-engine`.** On DS 9.0 / TRT 10 / CUDA 13 the `nvinfer` JIT path aborts with `NVRTC_ERROR_COMPILATION` when the ONNX embeds the custom `EfficientNMSX_TRT` / `ROIAlignX_TRT` ops. `trtexec` builds the same engine cleanly; `nvinfer` then only deserializes it.

### 4. Run the pipeline
```bash
# Pure VLM (no grounding)
python3 main.py assets/videos/sample.mp4 \
    -c configs/config_driving_scene.yaml \
    --output results/out.json \
    --kafka-bootstrap localhost:9092 --topic vlm-results

# YOLOE detection → VLM grounded on open-vocab classes
python3 main.py assets/videos/sample.mp4 \
    -c configs/config_driving_scene.yaml --detect \
    --detect-config configs/config_infer_yolo26e.txt \
    --output results/out.json

# YOLOE seg → grounding + annotated MP4
python3 main.py assets/videos/sample.mp4 \
    -c configs/config_driving_scene.yaml --detect \
    --detect-config configs/config_infer_yolo26e_seg.txt \
    --detect-output results/seg_osd.mp4 \
    --output results/out.json
```

Add `--dry-run` to skip Kafka and print results to stdout.

| Flag | Meaning |
|---|---|
| `-c`, `--config` | YAML config (model + prompts + segment window) |
| `--detect` | Enable object-detection branch |
| `--detect-config` | `nvinfer` `.txt` path; default = `configs/config_infer_yolo26.txt` |
| `--detect-output` | Write an OSD MP4 (bbox + instance masks for seg configs) |
| `--output` | Save all segments as a JSON array |
| `--kafka-bootstrap`, `--topic` | Kafka producer target |
| `--dry-run` | Skip Kafka |

---

## Why open-vocab matters

YOLOE bakes arbitrary class-name embeddings into the detector head at export time (`model.set_classes(...)` → `head.fuse(txt_pe)`). Ship a driving-scene model with `{vehicle, pedestrian, traffic_sign}` or a retail-camera model with `{shelf, cart, person}` from the same checkpoint — no retraining, no runtime text-prompt cost. Detection hints the VLM receives match **your** vocabulary instead of COCO-80.

In My-Curator, this becomes the **Scout's symbolic grounding**: YOLOE inventory is aggregated per clip (lateral zone + proximity for VRUs) and injected into the VLM prompt so Cosmos-Reason2 can sanity-check its own captions. Phase 2 of the expansion adds a separate **Judge** that cross-checks the DNA output against the inventory and flags `pedestrian_not_grounded`-style hallucinations.

---

## Project layout (current)

```
My-Curator/
├── main.py                               # entrypoint
├── docker-compose.yml                    # baseline: ds9-vlm-dev + Kafka + Zookeeper
├── plugin/
│   ├── gstnvvllmvlm.py                   # GStreamer VLM element (nvvllmvlm)
│   ├── vlm_utils.py                      # pure utils (host-testable)
│   ├── config_loader.py                  # YAML config singleton
│   └── output_schema.py                  # DrivingSceneResult (Pydantic) — Phase-1 replaced by Scenario DNA
├── src/
│   ├── vllm_ds_app_kafka_publish.py      # pipeline builder + Kafka + OSD branch
│   └── consumer.py                       # optional Kafka consumer
├── scripts/                              # YOLO26 / YOLOE download + ONNX export
├── configs/                              # nvinfer .txt + YAML prompts
├── lib/                                  # DS 9.0 custom YOLO parsers (.so)
├── docs/                                 # planning + context docs
│   ├── implementation_plan.md            # self-contained PR-level plan
│   └── implementation_plan_KOR.md        # Korean mirror
├── .github/                              # issue/PR templates + auto-assign + CODEOWNERS
└── tests/
    ├── unit/                             # 148 tests — host-runnable
    ├── integration/                      # 73 tests — GStreamer auto-mocked
    └── e2e/                               #   5 tests — Docker + GPU
```

---

## Tests

Baseline repository ships **238 test functions** (pre-P1-1 count, confirmed). All must stay green while the expansion lands.

```bash
# host — unit + integration, no GPU/Docker needed
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python pytest pytest-mock pytest-cov PyYAML pydantic
.venv/bin/pytest tests/unit tests/integration -q

# full suite inside Docker
pytest tests/ -v
```

The expansion adds new pytest markers — `schema`, `performance`, `simulation`, `prompt_regression`, `gpu`, `slow`. These are promoted to the repo-root `pytest.ini` in PR `P1-1`.

Lint: `ruff check . && ruff format --check .` (config in `pyproject.toml`).

---

## Contributing

- Issues / PRs are auto-labeled and auto-assigned — see `.github/`.
- PR title convention: `[P{phase}-{n}] summary` (e.g. `[P2-3] Best-of-N aggregator with YOLO26 symbolic reward`).
- Do not bump `vllm==0.14.0` — the FP8 Cosmos checkpoint produces gibberish on 0.15.1+.

---

## Tech stack

| Component | Version | Role |
|---|---|---|
| DeepStream | 9.0 | GStreamer GPU pipeline |
| Cosmos-Reason2-8B | FP8 | Scout VLM (captioner) |
| vLLM | **0.14.0** (pinned) | VLM inference engine |
| YOLO26 | m/s/l | Closed-vocab detector |
| YOLOE-26 seg | m/s/l | Open-vocab detector + segmentor |
| TensorRT | 10.14 | `nvinfer` backend |
| CUDA | 13.1 | Toolchain |
| Kafka | 7.6 | Result publishing |
| *Milvus (GPU_CAGRA)* | 2.6 | *Phase 1 — vector search* |
| *PostgreSQL (JSONB+GIN)* | 17 | *Phase 1 — DNA store* |
| *MinIO* | 2026.01 | *Phase 1 — blob store* |
| *Qwen2.5-14B-AWQ* | — | *Phase 2 — Judge LLM* |
| *CARLA* | 0.9.15 | *Phase 4 — synthetic corner cases* |

Italicized rows are planned additions.

## References

- [NVIDIA DeepStream SDK 9.0](https://developer.nvidia.com/deepstream-sdk)
- [deepstream-vllm-plugin](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-vllm-plugin) — upstream `nvvllmvlm`
- [Cosmos-Reason2-8B-FP8](https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/)
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) / [DeepStream-Yolo-Seg](https://github.com/marcoslucianops/DeepStream-Yolo-Seg)
- [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)
- [Semantic-Drive](https://github.com/AntonioAlgaida/Semantic-Drive) — 4-layer ontology inspiration
- [NVIDIA Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate) — reference AV curation pipeline

## Attribution

This project is a **fork-and-extend** of [DeepStream-VLM](https://github.com/IJLee0812/DeepStream-VLM). The baseline GStreamer pipeline, `nvvllmvlm` integration, YOLO/YOLOE export toolchain, Kafka publisher, and test scaffolding all carry over from the upstream repo; My-Curator layers a curation/validation/search/simulation stack on top.

## License

Apache 2.0
