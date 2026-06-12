#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/csteinhauer/robot_vision"
ASTRA_WS="/home/csteinhauer/astra_ws"
HOST="0.0.0.0"
PORT="8080"
URL="http://127.0.0.1:${PORT}"
LAN_URL="http://jetson.local:${PORT}"
LOG_DIR="${APP_DIR}/data/logs"
LOG_FILE="${LOG_DIR}/robot_vision_web.log"
ASTRA_LOG="${LOG_DIR}/astra_camera.log"
BROWSER_LOG="${LOG_DIR}/browser_open.log"

open_robot_vision_url() {
  local url="$1"
  local opened=1

  {
    echo "Browser open attempt: $(date)"
    echo "URL=${url}"
    echo "Initial DISPLAY=${DISPLAY:-}"
    echo "Initial XAUTHORITY=${XAUTHORITY:-}"
    echo "Initial DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-}"
    echo "Available browser commands:"
    for command_name in chromium-browser chromium google-chrome firefox xdg-open gio; do
      if command -v "${command_name}" >/dev/null 2>&1; then
        echo "  ${command_name}: $(command -v "${command_name}")"
      else
        echo "  ${command_name}: missing"
      fi
    done
  } >> "${BROWSER_LOG}"

  if [ -z "${DISPLAY:-}" ]; then
    if [ -S /tmp/.X11-unix/X0 ]; then
      export DISPLAY=":0"
    fi
  fi

  if [ -z "${XAUTHORITY:-}" ] && [ -f "/home/csteinhauer/.Xauthority" ]; then
    export XAUTHORITY="/home/csteinhauer/.Xauthority"
  fi

  if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] && [ -S "/run/user/$(id -u)/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"
  fi

  if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  fi

  if [ -n "${DISPLAY:-}" ]; then
    if command -v chromium-browser >/dev/null 2>&1; then
      echo "Launching chromium-browser --new-window ${url}" >> "${BROWSER_LOG}"
      nohup chromium-browser --new-window "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening Chromium: ${url}"
      opened=0
    fi
    if command -v chromium >/dev/null 2>&1; then
      echo "Launching chromium --new-window ${url}" >> "${BROWSER_LOG}"
      nohup chromium --new-window "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening Chromium: ${url}"
      opened=0
    fi
    if command -v google-chrome >/dev/null 2>&1; then
      echo "Launching google-chrome --new-window ${url}" >> "${BROWSER_LOG}"
      nohup google-chrome --new-window "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening Chrome: ${url}"
      opened=0
    fi
    if command -v xdg-open >/dev/null 2>&1; then
      echo "Launching xdg-open ${url}" >> "${BROWSER_LOG}"
      nohup xdg-open "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening browser: ${url}"
      opened=0
    fi
    if command -v gio >/dev/null 2>&1; then
      echo "Launching gio open ${url}" >> "${BROWSER_LOG}"
      nohup gio open "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening browser with gio: ${url}"
      opened=0
    fi
    if command -v firefox >/dev/null 2>&1; then
      echo "Launching firefox --new-window ${url}" >> "${BROWSER_LOG}"
      nohup firefox --new-window "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening Firefox: ${url}"
      opened=0
    fi
    if command -v python3 >/dev/null 2>&1; then
      echo "Launching python3 -m webbrowser -t ${url}" >> "${BROWSER_LOG}"
      nohup python3 -m webbrowser -t "${url}" >> "${BROWSER_LOG}" 2>&1 &
      echo "Opening browser with python webbrowser: ${url}"
      opened=0
    fi
  fi

  if [ "${opened}" -ne 0 ]; then
    echo "Could not auto-open a browser; open ${url} manually."
    echo "Browser open failed: no usable DISPLAY/browser command." >> "${BROWSER_LOG}"
    return 1
  fi
  return 0
}

mkdir -p "${LOG_DIR}"
cd "${APP_DIR}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
if [ -f "/opt/ros/humble/setup.bash" ]; then
  set +u
  source /opt/ros/humble/setup.bash
  if [ -f "${ASTRA_WS}/install/setup.bash" ]; then
    source "${ASTRA_WS}/install/setup.bash"
  fi
  set -u
fi
CAMERA_PROVIDER="$(
  python3 - <<'PY' 2>/dev/null || true
import yaml
with open("config/app.yaml", "r", encoding="utf-8") as handle:
    raw = yaml.safe_load(handle) or {}
print(str((raw.get("camera") or {}).get("provider", "")).strip().lower())
PY
)"
CAMERA_PROVIDER="${CAMERA_PROVIDER:-orbbec}"

echo "Robot Vision startup"
echo "App: ${APP_DIR}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Camera provider=${CAMERA_PROVIDER}"
echo "URL: ${LAN_URL}"

if pgrep -f "python3 -m robot_vision" >/dev/null 2>&1; then
  echo "Stopping existing Robot Vision process..."
  pgrep -f "python3 -m robot_vision" | xargs -r kill
  sleep 1
fi

if [[ "${CAMERA_PROVIDER}" == "ros_astra" || "${CAMERA_PROVIDER}" == "astra_ros" || "${CAMERA_PROVIDER}" == "ros" ]]; then
  if ps -ef | grep -E "ros2 launch astra_camera|astra_camera_container" | grep -v grep >/dev/null 2>&1; then
    echo "Stopping existing Astra ROS camera process..."
    ps -ef | awk '/ros2 launch astra_camera|astra_camera_container/ && !/awk/ {print $2}' | xargs -r kill
    sleep 2
  fi

  echo "Starting Astra ROS camera driver..."
  set +u
  source /opt/ros/humble/setup.bash
  source "${ASTRA_WS}/install/setup.bash"
  set -u
  nohup ros2 launch astra_camera astra.launch.py > "${ASTRA_LOG}" 2>&1 < /dev/null &
  astra_pid="$!"
  echo "Started Astra ROS PID ${astra_pid}"

  for attempt in {1..30}; do
    if timeout 2 ros2 topic echo /camera/depth/image_raw --once --field header >/dev/null 2>&1; then
      echo "Astra depth topic is ready."
      break
    fi
    if [ "${attempt}" -eq 30 ]; then
      echo "Astra depth topic did not become ready. Last Astra log lines:"
      tail -n 80 "${ASTRA_LOG}" || true
      exit 1
    fi
    sleep 0.5
  done
else
  echo "Skipping pre-start Astra ROS camera driver for provider ${CAMERA_PROVIDER}."
  if ps -ef | grep -E "ros2 launch astra_camera|astra_camera_container" | grep -v grep >/dev/null 2>&1; then
    echo "Stopping existing Astra ROS camera process..."
    ps -ef | awk '/ros2 launch astra_camera|astra_camera_container/ && !/awk/ {print $2}' | xargs -r kill
    sleep 1
  fi
fi

echo "Starting Robot Vision server..."
nohup python3 -m robot_vision \
  --config config/app.yaml \
  --host "${HOST}" \
  --port "${PORT}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

server_pid="$!"
echo "Started PID ${server_pid}"

for attempt in {1..20}; do
  if python3 - <<PY >/dev/null 2>&1
import json
import urllib.request
with urllib.request.urlopen("${URL}/api/health", timeout=1.5) as response:
    payload = json.loads(response.read().decode("utf-8"))
if not payload.get("ok"):
    raise SystemExit(1)
PY
  then
    echo "Robot Vision is ready: ${LAN_URL}"
    open_robot_vision_url "${URL}" || true
    exit 0
  fi
  sleep 0.5
done

echo "Robot Vision did not become ready. Last log lines:"
tail -n 40 "${LOG_FILE}" || true
exit 1
