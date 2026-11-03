#!/usr/bin/env bash
# Stop the ranks, save one gzipped archive with their logs and keep the newest LOG_KEEP (10).
# Usage: ./stop.sh; DRY_RUN=1 ./stop.sh prints the plan and changes nothing.
# Only this recipe's container (label tf.recipe=glm53-zig) is stopped; any other container is left alone.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ && "$LOG_KEEP" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG 'STOP_TIMEOUT and LOG_KEEP must be positive integers.'
[[ "$CONTAINER_NAME" != glm53-tf ]] || fail E_CONFIG 'glm53-tf is the Python recipe'"'"'s container; this script never stops it.'
# A warning only. /health's live.connections counts the requests running and waiting; a count it cannot read is not none.
warn_open_requests() {
  local host="$HOST" open
  [[ "$host" != 0.0.0.0 && "$host" != :: ]] || host=127.0.0.1
  [[ "$host" != *:* ]] || host="[$host]"
  open=$(curl -s --max-time 3 -w '\n%{http_code}' "http://$host:$PORT/health" 2>/dev/null | python3 -c '
import json, sys
body, _, code = sys.stdin.read().rpartition("\n")
r = json.loads(body) if code == "200" else {}
n = r["live"]["connections"] if type(r) is dict and r.get("error") is None else None
sys.exit(1) if type(n) is not int or n < 0 else print(n)' 2>/dev/null) || open='?'
  if [[ "$open" == '?' ]]; then
    warn W_OPEN 'Could not read the open requests from /health; any still open get TF_DRAIN_S to finish, then are cut off.'
  elif (( open > 0 )); then
    warn W_OPEN "$open request(s) open; they get TF_DRAIN_S to finish, then are cut off."
  fi
}
if [[ "$DRY_RUN" == 1 ]]; then
  log "Stop $CONTAINER_NAME only if it carries the label tf.recipe=glm53-zig (rank 0 drains first, up to $STOP_TIMEOUT s):"
  plan docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME"
  log "  Save the rank logs and Docker output as $LOG_DIR/glm53-zig-<timestamp>.tar.gz; keep the newest $LOG_KEEP archives."
  plan docker rm "$CONTAINER_NAME"
  exit 0
fi
command -v docker >/dev/null || fail E_DEPENDENCY 'Missing command: docker.'
docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
if [[ "${GLM_STOP_LOCK_HELD:-0}" != 1 ]]; then
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
fi
if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  log "No container named $CONTAINER_NAME. Nothing to stop."; exit 0
fi
[[ "$(docker inspect -f '{{index .Config.Labels "tf.recipe"}}' "$CONTAINER_NAME")" == glm53-zig ]] \
  || fail E_STOP "$CONTAINER_NAME was not started by this recipe (no tf.recipe=glm53-zig label); not touching it."
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != true ]] || warn_open_requests
log "Stopping $CONTAINER_NAME (up to $STOP_TIMEOUT s; requests in progress get TF_DRAIN_S to finish)."
docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]; then
  warn W_STOP "Container still runs after the ${STOP_TIMEOUT}s stop; docker kill."
  docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]; then
  fail E_STOP "Container still runs after stop and kill. A human must clean it up: docker ps; docker rm -f $CONTAINER_NAME."
fi
run_logs=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/logs"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")
if [[ -z "$run_logs" || ! -d "$run_logs" ]]; then
  fail E_LOG 'Cannot locate the rank log directory. The stopped container was retained; inspect its mounts before removing it.'
fi
mkdir -p "$LOG_DIR"
archive="$LOG_DIR/glm53-zig-$(date -u +%Y%m%dT%H%M%SZ)-$$.tar.gz"
docker logs --timestamps "$CONTAINER_NAME" > "$run_logs/container.log" 2>&1 || fail E_LOG 'Could not save Docker output. The stopped container was retained.'
docker inspect "$CONTAINER_NAME" > "$run_logs/container.json" 2>/dev/null || true
tar -czf "$archive.tmp" -C "$run_logs" . || fail E_LOG 'Could not archive logs. Check LOG_DIR space and permissions; the stopped container was retained.'
mv -- "$archive.tmp" "$archive"
docker rm "$CONTAINER_NAME" >/dev/null || fail E_STOP 'Logs were saved, but Docker could not remove the stopped container.'
# Raw logs are deleted only after a successful archive and only in a recipe-owned run directory.
if [[ "$run_logs" == "$LOG_DIR"/run-* && ! -L "$run_logs" ]]; then
  rm -rf -- "$run_logs"
  [[ ! -L "$LOG_DIR/current" || "$(readlink -- "$LOG_DIR/current")" != "$run_logs" ]] || rm -- "$LOG_DIR/current"
fi
python3 - "$LOG_DIR" "$LOG_KEEP" <<'PY'
from pathlib import Path
import sys
archives = sorted(Path(sys.argv[1]).glob("glm53-zig-*.tar.gz"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
for path in archives[int(sys.argv[2]):]:
    path.unlink()
PY
log "Stopped. Logs saved to $archive (newest $LOG_KEEP kept)."
