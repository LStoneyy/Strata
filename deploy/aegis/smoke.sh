#!/usr/bin/env bash
# One smoke run on aegis: start serve/server.py with a run config, watch VRAM/GTT/RAM, send the fixed requests, stop.
#   deploy/aegis/smoke.sh <config.json> <name>          results in ~/strata-data/smoke/<name>/
#   LOAD_TIMEOUT=1200 ...                                seconds to wait for "loaded" (first start reads ~25 GB)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG=$(readlink -f "$1"); NAME=$2
OUT="$HOME/strata-data/smoke/$NAME"; mkdir -p "$OUT"; : > "$OUT/results.jsonl"
PORT=${PORT:-8012}; LOAD_TIMEOUT=${LOAD_TIMEOUT:-1200}
cp "$CFG" "$OUT/config.json"
journalctl -k --since now -f --no-pager > "$OUT/kernel.log" 2>/dev/null & JPID=$!

cd "$ROOT"
setsid .venv/bin/python serve/server.py --engine strata --config "$CFG" --port "$PORT" > "$OUT/server.log" 2>&1 < /dev/null &
SRV=$!
echo "server pid $SRV, config $CFG" | tee "$OUT/run.txt"
"$ROOT/deploy/aegis/watchdog.sh" "$SRV" "$OUT" & WD=$!

t0=$(date +%s); ok=""
while kill -0 "$SRV" 2>/dev/null; do
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"loaded": *true'; then ok=1; break; fi
  (( $(date +%s) - t0 > LOAD_TIMEOUT )) && break
  sleep 2
done
echo "load: $(( $(date +%s) - t0 )) s, ok=${ok:-no}" | tee -a "$OUT/run.txt"

if [[ -n "$ok" ]]; then
  .venv/bin/python deploy/aegis/smoke.py "http://127.0.0.1:$PORT" "$OUT" --long-file "$ROOT/serve/split.py" | tee -a "$OUT/run.txt"
fi

# stop: SIGTERM takes Ctrl+C's path (QUIT to the engine); kill -9 the tree if it hangs
kill -TERM "$SRV" 2>/dev/null
for _ in $(seq 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
if kill -0 "$SRV" 2>/dev/null; then
  echo "server did not stop in 60 s: kill -9" | tee -a "$OUT/run.txt"
  pkill -9 -g "$SRV" 2>/dev/null; kill -9 "$SRV" 2>/dev/null
fi
pkill -9 -f "[e]ngine/strata " 2>/dev/null
wait "$WD" 2>/dev/null
kill "$JPID" 2>/dev/null
tail -1 "$OUT/watch.log" | tee -a "$OUT/run.txt"
grep -E "^(ABORT|WARN)" "$OUT/watch.log" | tee -a "$OUT/run.txt"
grep -iE "ttm|amdgpu.*(error|warn|fault)|BUG|general protection" "$OUT/kernel.log" | head -5 | tee -a "$OUT/run.txt"
echo "done: $OUT"
