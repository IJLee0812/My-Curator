#!/bin/bash
# CARLA 0.9.15 launcher (P5-1).
#
#   CARLA_VIEWER      1 = Xvfb + x11vnc + noVNC, 0 = -RenderOffScreen (default 1)
#   CARLA_QUALITY     UE4 quality level (default Low)
#   CARLA_RESOLUTION  Xvfb screen size, viewer only (default 800x600)
#   CARLA_MAP         town to boot on, e.g. Town03 (default: the engine's own default)
#
# CARLA_MAP exists because switching maps at runtime segfaults this build: a plain
# client.load_world("Town03") takes the server down with signal 11, with no client
# library and no scenario involved. Boot the server on the map you need instead.

VIEWER="${CARLA_VIEWER:-1}"
QUALITY="${CARLA_QUALITY:-Low}"
RESOLUTION="${CARLA_RESOLUTION:-800x600}"
MAP="${CARLA_MAP:-}"

# The startup map comes from the engine config, not the command line: this packaged build
# silently ignores a map passed as an argument (measured — `CarlaUE4 /Game/Carla/Maps/Town03`
# still comes up on the configured default), and switching at runtime segfaults it.
ENGINE_INI="/home/carla/CarlaUE4/Config/DefaultEngine.ini"
if [ -n "$MAP" ]; then
    if [ -w "$ENGINE_INI" ]; then
        sed -i -E "s#^(GameDefaultMap|ServerDefaultMap)=.*#\\1=/Game/Carla/Maps/$MAP.$MAP#" \
            "$ENGINE_INI"
        echo "[carla-entrypoint] booting on map $MAP"
    else
        echo "[carla-entrypoint] WARNING: $ENGINE_INI is not writable; CARLA_MAP ignored" >&2
    fi
fi

# The UE4 renderer needs Vulkan; without a working ICD it exits at startup.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

if [ "$VIEWER" != "1" ]; then
    echo "[carla-entrypoint] headless, RPC on :2000"
    exec ./CarlaUE4.sh -nosound -RenderOffScreen -carla-rpc-port=2000 \
        -quality-level="$QUALITY" "$@"
fi

rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
mkdir -p /tmp/.X11-unix 2>/dev/null || true
chmod 1777 /tmp/.X11-unix 2>/dev/null || true

Xvfb :99 -screen 0 "${RESOLUTION}x24" -nolisten tcp &
export DISPLAY=:99
sleep 2

x11vnc -display :99 -forever -nopw -rfbport 5900 -shared -noxdamage >/dev/null 2>&1 &
sleep 1

websockify --web=/usr/share/novnc/ 6080 localhost:5900 &

echo "[carla-entrypoint] viewer on :6080, RPC on :2000"
exec ./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -nosound \
    -carla-rpc-port=2000 -quality-level="$QUALITY" "$@"
