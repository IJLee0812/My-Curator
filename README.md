# My-Curator

An autonomous driving front-camera **clip curation and validation platform**
built on NVIDIA DeepStream 9.0. A GPU pipeline segments driving video, annotates
each clip with a schema-validated **Scenario DNA v0.1** record, and surfaces the
corpus through a hybrid vector+JSONB search API and a React curation console.

**Live:** [ijlee0812.github.io/My-Curator](https://ijlee0812.github.io/My-Curator)

---

## Architecture

![Architecture](assets/images/architecture.png)

---

## Tech stack

| Component | Version | Role |
|---|---|---|
| DeepStream | 9.0 | GStreamer GPU pipeline |
| Cosmos-Reason2-8B | FP8 | Scout VLM |
| Cosmos-Embed1-336p | — | Video embedding (768-dim) |
| vLLM | **0.14.0** (pinned) | VLM inference engine |
| YOLO26 | m / s / l | Closed-vocab detector |
| TensorRT | 10.14 | `nvinfer` backend |
| CUDA | 13.1 | Toolchain |
| FastAPI | — | curation-api (port 8001) |
| Next.js | 16.2.6 + React 19 | Curation console (port 3000) |
| Kafka | 7.6 | Event bus |
| Milvus (GPU_CAGRA) | 2.6.15 | Vector store (IP metric) |
| PostgreSQL | 17 | DNA store (JSONB + GIN) |
| MinIO | — | Object store |

---

## Project layout

```
My-Curator/
├── my_curator/                   # clean-architecture package (pip install -e .)
│   ├── domain/                   # Scout protocol · DNA validator · timestamp parser
│   ├── adapters/                 # PG · Milvus · MinIO · streaming · Cosmos-Embed1 · GStreamer
│   ├── application/              # pipeline · curation consumer · embedder worker
│   ├── interfaces/               # FastAPI app + routers (search · ingest · clips · review)
│   └── cli/                      # run_pipeline · run_curation_consumer · run_embedder
├── services/ui/                  # Next.js 16 curation console
├── infra/
│   ├── compose.base.yml          # storage stack — Postgres + Milvus + MinIO + etcd
│   ├── compose.curate.yml        # curate overlay — Kafka + curation-api + embedder + UI
│   └── compose.pipeline.yml      # DS pipeline container
├── schemas/                      # scenario_dna_v0_1.schema.json (frozen)
├── prompts/                      # scout_cosmos_reason2.v1.md
├── configs/                      # nvinfer .txt configs + YAML pipeline config
└── tests/                        # unit · integration · e2e
```

---

## Quickstart

**Prerequisites:** Docker + NVIDIA Container Toolkit · driver 580+ · NGC API key ·  
2× RTX 4090-class GPUs (GPU 0: curate stack · GPU 1: DS pipeline, ~20 GiB VRAM).

```bash
git clone https://github.com/IJLee0812/My-Curator.git
cd My-Curator
cp .env.example .env   # fill in PG_*, MINIO_*, NGC_API_KEY, VIDEO_DATA_ROOT
```

**Start the storage + curate stack:**

```bash
docker compose --env-file .env \
  -f infra/compose.base.yml \
  -f infra/compose.curate.yml up -d
```

- Curation console → `http://localhost:3000`
- curation-api → `http://localhost:8001`

**Run the DS pipeline:**

```bash
docker compose --env-file .env -f infra/compose.pipeline.yml run --rm \
  -e CUDA_VISIBLE_DEVICES=1 my-curator-ds9-vlm-dev \
  python3 -m my_curator.cli.run_pipeline <video.mp4> \
  -c configs/config_driving_scene.yaml [--dry-run]
```

---

## Tests

```bash
# unit + integration — no GPU or Docker required
.venv/bin/pytest tests/unit tests/integration -q
# → 661 passed, 3 skipped

# e2e — requires compose stack
.venv/bin/pytest tests/e2e -m e2e -v
# → 28 passed, 10 skipped
```

Lint: `ruff check . && ruff format --check .`

---

## References

- [NVIDIA DeepStream SDK 9.0](https://developer.nvidia.com/deepstream-sdk)
- [deepstream-vllm-plugin](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-vllm-plugin)
- [Cosmos-Reason2-8B-FP8](https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/)
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) / [DeepStream-Yolo-Seg](https://github.com/marcoslucianops/DeepStream-Yolo-Seg)
- [NVIDIA Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate)

## License

Apache 2.0
