FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Modern setuptools (>=64) for PEP 660 editable-install support.
RUN pip3 install --no-cache-dir --upgrade pip "setuptools>=68" wheel

# torch + torchvision first (large wheels, cached separately)
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# runtime deps pinned to match project versions
RUN pip3 install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "httpx>=0.28.0" \
    "asyncpg==0.31.0" \
    "pymilvus==2.6.12" \
    "kafka-python==2.3.1" \
    "boto3==1.42.96" \
    "pydantic==2.12.5" \
    "PyYAML==6.0.3" \
    "Pillow==12.2.0" \
    "transformers>=4.40.0,<5.0.0" \
    "accelerate>=0.30.0" \
    "einops>=0.7.0" \
    "jsonschema>=4.0.0"

# R-6 single-COPY: only the my_curator/ package + schemas + prompts.
# pyproject.toml comes along so ``pip install -e .`` finds the package
# discovery rules ([tool.setuptools.packages.find] include = ["my_curator*"]).
# Old src/ + services/ source trees are intentionally NOT copied — the
# shims at those paths are not needed inside this container after R-6.
COPY pyproject.toml ./
COPY my_curator/ ./my_curator/
COPY schemas/ ./schemas/
COPY prompts/ ./prompts/

RUN pip3 install --no-cache-dir .

EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "my_curator.interfaces.http.curation_api.app:app", \
     "--host", "0.0.0.0", "--port", "8001"]
