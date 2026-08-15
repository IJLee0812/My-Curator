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

COPY carla-entrypoint.sh /carla-entrypoint.sh
RUN chmod +x /carla-entrypoint.sh

USER carla
WORKDIR /home/carla
