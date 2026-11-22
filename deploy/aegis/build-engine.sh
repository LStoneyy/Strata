#!/usr/bin/env bash
# Build the Strata HIP engine for aegis: RX 5500 XT (gfx1012) + RX 9060 XT (gfx1200), Ubuntu's ROCm 7.1 under /usr.
#
# 0.2.16+ layout: engine sources in engine/, setup/server in terminal/ (setup's ROOT); the engine goes to terminal/engine/.
# Same switches as terminal/setup.py build_engine_hip(), but by hand: setup does not build gfx1012 (docs/OLDER_GPUS.md) and its
# rocm_root() expects /opt/rocm.  The engine is copied to engine/strata with a BUILD.json that setup.py accepts as
# "engine already built for this PC" (same source hash, backend hip, both archs), plus lib_dirs pointing at the rocBLAS
# with gfx1012 kernels (deploy/aegis/build-rocblas.sh).
#
#   deploy/aegis/build-engine.sh            configure + build + ctest + install into engine/
#   JOBS=4 deploy/aegis/build-engine.sh     fewer compile jobs (14 GiB RAM)
#   SKIP_TESTS=1 ...                        no ctest
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TERM_DIR="$ROOT/terminal"
cd "$TERM_DIR"
JOBS="${JOBS:-8}"
ARCHS="${ARCHS:-gfx1012;gfx1200}"
ROCBLAS_PREFIX="${ROCBLAS_PREFIX:-$HOME/opt/rocblas-7.1.0-gfx1012-gfx1200-merged}"   # merged: see build-rocblas.sh MERGED
BUILD="$TERM_DIR/build-hip"
PY="$TERM_DIR/.venv/bin/python"

[[ -f "$ROCBLAS_PREFIX/lib/librocblas.so.5" ]] || { echo "rocBLAS with gfx1012 missing: $ROCBLAS_PREFIX (run build-rocblas.sh)" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "no terminal/.venv: cd terminal && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2; exit 1; }

cmake -S "$ROOT/engine" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS="$([[ ${SKIP_TESTS:-0} == 1 ]] && echo OFF || echo ON)" \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="$ARCHS" \
  -DCMAKE_HIP_COMPILER=/usr/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/usr

nice -n 10 cmake --build "$BUILD" -j "$JOBS"

if [[ ${SKIP_TESTS:-0} != 1 ]]; then
  # the GPU tests run against the rocBLAS that has gfx1012 kernels, like the engine will
  LD_LIBRARY_PATH="$ROCBLAS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ctest --test-dir "$BUILD" --output-on-failure -j 4 2>&1 | tee "$BUILD/ctest.log" | tail -25 || true
fi

mkdir -p engine
cp -f "$BUILD/strata" engine/strata
# BUILD.json as setup.py's build_engine_hip() writes it, so setup keeps this engine instead of building one
"$PY" - "$ARCHS" "$ROCBLAS_PREFIX/lib" <<'EOF'
import json, sys
from pathlib import Path
archs = sorted(set(a for a in sys.argv[1].replace(",", ";").split(";") if a))
lib = sys.argv[2]
import setup                                    # setup.py: module-level code only defines things; main() is guarded
meta = {"source": "local-hip", "backend": "hip", "version": setup.source_version(), "archs": archs, "vision": "none",
        "lib_dirs": [lib], "src": setup.source_hash(setup.ENGINE_SOURCES)}
floor = setup.cpu_floor(setup.cpu_info()[1])
if floor:
    meta["isa_floor"] = floor
Path("engine/BUILD.json").write_text(json.dumps(meta, indent=1))
print("engine/BUILD.json:", json.dumps(meta))
EOF
{
  echo "Strata engine $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty'), archs $ARCHS"
  echo "built $(date -Is) on $(hostname), $(/usr/bin/clang++ --version | head -1)"
  echo "rocBLAS: $ROCBLAS_PREFIX"
} > engine/BUILD-INFO.txt
echo "engine: $TERM_DIR/engine/strata"
