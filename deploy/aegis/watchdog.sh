#!/usr/bin/env bash
# VRAM/GTT/RAM watchdog for a Strata server on aegis (after ~/llama.cpp.vulkan-20261105/smoke/watchdog.sh).
#   card0 = RX 5500 XT (8176 MiB), card1 = RX 9060 XT (16304 MiB)   (DRM numbering; HIP numbers them the same way here)
# Kills the server's process tree (kill -9) when
#   - a card has less than MINFREE MiB VRAM free
#   - GTT of both cards together exceeds GTT_LIMIT MiB
#   - eviction: a card's VRAM falls >500 MiB under its peak AND its GTT rises >300 MiB over the value at the peak
#   - MemAvailable falls under MINRAM MiB
# Logs one line every 0.25 s to $OUT/watch.log; WARN once when a card's headroom is under 1 GiB.
#   watchdog.sh <server-pid> <outdir>
SRV=$1; OUT=$2
MINFREE=${MINFREE:-256}; GTT_LIMIT=${GTT_LIMIT:-6800}; MINRAM=${MINRAM:-400}
T0=8176; T1=16304
D=/sys/class/drm
: > "$OUT/watch.log"
kill_tree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done; kill -9 "$p" 2>/dev/null; }
p0=0; p1=0; gp0=0; gp1=0; warned=""; minram=999999; maxswap=0
while kill -0 "$SRV" 2>/dev/null; do
  # an unreadable counter (driver trouble) aborts the run instead of leaving the checks below without numbers
  if ! { v0=$(<$D/card0/device/mem_info_vram_used) && g0=$(<$D/card0/device/mem_info_gtt_used) &&
         v1=$(<$D/card1/device/mem_info_vram_used) && g1=$(<$D/card1/device/mem_info_gtt_used); } 2>/dev/null ||
     [[ ! "$v0$g0$v1$g1" =~ ^[0-9]+$ ]]; then
    echo "ABORT GPU memory counters unreadable" >> "$OUT/watch.log"; kill_tree "$SRV"; break
  fi
  v0=$(( v0/1048576 )); g0=$(( g0/1048576 )); v1=$(( v1/1048576 )); g1=$(( g1/1048576 ))
  read -r ma st sf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int(a/1024),int(t/1024),int(f/1024)}' /proc/meminfo)
  sw=$(( st - sf ))
  (( v0 > p0 )) && { p0=$v0; gp0=$g0; }
  (( v1 > p1 )) && { p1=$v1; gp1=$g1; }
  (( ma < minram )) && minram=$ma
  (( sw > maxswap )) && maxswap=$sw
  echo "$(date +%T.%N | cut -c1-12) v0=$v0 g0=$g0 v1=$v1 g1=$g1 avail=$ma swap=$sw" >> "$OUT/watch.log"
  reason=""
  (( T0 - v0 < MINFREE )) && reason="5500 free $((T0-v0)) MiB"
  (( T1 - v1 < MINFREE )) && reason="9060 free $((T1-v1)) MiB"
  (( g0 + g1 > GTT_LIMIT )) && reason="GTT total $((g0+g1)) MiB"
  (( ma < MINRAM )) && reason="MemAvailable $ma MiB"
  (( v0 < p0 - 500 && g0 > gp0 + 300 )) && reason="eviction 5500 v0=$v0 peak=$p0 g0=$g0"
  (( v1 < p1 - 500 && g1 > gp1 + 300 )) && reason="eviction 9060 v1=$v1 peak=$p1 g1=$g1"
  if [[ -n "$reason" ]]; then
    echo "ABORT $reason" >> "$OUT/watch.log"; kill_tree "$SRV"; break
  fi
  if [[ -z "$warned" ]] && (( T0 - v0 < 1024 || T1 - v1 < 1024 )); then
    echo "WARN headroom < 1 GiB: 5500 free $((T0-v0)), 9060 free $((T1-v1))" >> "$OUT/watch.log"; warned=1
  fi
  sleep 0.25
done
echo "PEAK v0=$p0/$T0 (free $((T0-p0))) v1=$p1/$T1 (free $((T1-p1))) min_avail=$minram max_swap=$maxswap" >> "$OUT/watch.log"
