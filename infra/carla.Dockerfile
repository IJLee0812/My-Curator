# CARLA 0.9.15 simulation server (P5-1).
#
# Build context is infra/ — see compose.simulate.yml.
#   docker compose -f infra/compose.base.yml -f infra/compose.simulate.yml \
#     --env-file .env --profile simulate build carla-server

FROM carlasim/carla:0.9.15

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    xdg-user-dirs \
    x11-utils \
    libjpeg8 \
    libtiff5 \
    python3.7 \
    python3-pip \
    x11vnc \
    novnc \
    python-websockify \
    && rm -rf /var/lib/apt/lists/*

RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install \
        /home/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl

# scenario_runner (P5-3) — reads the .xosc files we generate, to prove a standard
# OpenSCENARIO tool accepts them. It is NOT the render path; nothing here drives a
# simulation. Fetched as a release tarball because the image has wget but no git.
RUN wget -qO /tmp/sr.tar.gz \
        https://github.com/carla-simulator/scenario_runner/archive/refs/tags/v0.9.15.tar.gz && \
    tar -xzf /tmp/sr.tar.gz -C /opt && \
    mv /opt/scenario_runner-0.9.15 /opt/scenario_runner && \
    rm /tmp/sr.tar.gz

# ephem is pinned to 4.2 because the image ships no compiler and no Python headers: every
# other release builds from source here and fails. 4.2 publishes a cp37 wheel, which
# avoids pulling a ~200 MB toolchain in for one dependency. Nothing else is pinned beyond
# scenario_runner's own requirements.txt, and --only-binary must NOT be used globally —
# networkx 2.2 ships as an sdist only (pure Python, so it needs no compiler).
RUN python3.7 -m pip install --no-cache-dir \
        py-trees==0.8.3 numpy==1.18.4 networkx==2.2 Shapely==1.7.1 psutil \
        ephem==4.2 xmlschema==1.0.18 tabulate opencv-python==4.2.0.32 \
        matplotlib six simple-watchdog-timer antlr4-python3-runtime==4.10 graphviz

# GStreamer (P5-4) — the render path. All video work happens in this image: the host has
# base+good plugins only, so it can neither encode nor decode H.264. gstreamer1.0-x carries
# textoverlay (the pango plugin) and gstreamer1.0-libav the decoder. ~135 MB installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-x \
    && rm -rf /var/lib/apt/lists/*

# scenario_runner imports CARLA's own agents package, which is not on the wheel's path;
# /opt/my-curator is where the render code is bind-mounted.
ENV PYTHONPATH=/opt/scenario_runner:/home/carla/PythonAPI/carla:/opt/my-curator

COPY carla-entrypoint.sh /carla-entrypoint.sh
RUN chmod +x /carla-entrypoint.sh

USER carla
WORKDIR /home/carla
