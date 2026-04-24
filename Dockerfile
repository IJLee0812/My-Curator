FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# System packages
#   - python3-gi + python3-gst-1.0 + gstreamer1.0-python3-plugin-loader:
#       Python GStreamer bindings required by plugin/gstnvvllmvlm.py
#   - ffmpeg + fonts-dejavu-core:
#       used by scripts/gen_desc_visuals.py and ad-hoc frame extraction
#   - build-essential:
#       needed when (re)building lib/libnvdsinfer_custom_impl_Yolo*.so
#       against the container's TRT 10 / CUDA 13 toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-gi \
      python3-gst-1.0 \
      gstreamer1.0-python3-plugin-loader \
      ffmpeg \
      fonts-dejavu-core \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies — single source of truth is requirements.txt.
# --ignore-installed lets vLLM override the DeepStream base image's older
# pip-installed vllm without breaking the image's other system packages.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed -r /tmp/requirements.txt

# NGC CLI (for `ngc registry model download-version` of the FP8 VLM)
RUN cd /tmp && \
    wget -q --content-disposition \
      'https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.63.0/files/ngccli_linux.zip' \
      -O ngc_cli.zip && \
    unzip -q ngc_cli.zip -d /usr/local/ && \
    chmod +x /usr/local/ngc-cli/ngc && \
    ln -sf /usr/local/ngc-cli/ngc /usr/local/bin/ngc && \
    rm ngc_cli.zip

WORKDIR /workspace
