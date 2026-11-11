#!/usr/bin/env bash
# Build rocBLAS 7.1.0 (library version 5.1.0) for gfx1012 + gfx1200 into a private prefix.
#
# Why: Ubuntu 26.04's librocblas5 7.1.0 is built without gfx1012, so hipblasCreate aborts on the RX 5500 XT
# ("Cannot read TensileLibrary.dat ... for GPU arch : gfx1012"). Upstream rocBLAS 7.1 lists gfx1012
# (TARGET_LIST_ROCM_7.1); Tensile maps it to navi14 and uses the hip fallback logic.
#
# Same sources and patches as the Ubuntu package (archive.ubuntu.com pool/universe/r/rocblas, 7.1.0-1ubuntu4), same
# Tensile options as debian/rules; only GPU_TARGETS, the prefix and the test/client switches differ. The process that
# needs it (Strata) runs with LD_LIBRARY_PATH=$PREFIX/lib, so the system hipBLAS loads this librocblas.so.5.
#
# Build deps (sudo apt install): rocm-cmake libblas-dev libmsgpack-cxx-dev clang clang-tools pkg-rocm-tools python3-joblib
set -euo pipefail

VER=7.1.0
DEBREV=1ubuntu4
TARGETS="${TARGETS:-gfx1012;gfx1200}"
JOBS="${JOBS:-8}"                                   # 14 GiB RAM: hipcc/Tensile jobs need ~1-1.5 GiB each
SRC="${SRC:-$HOME/src/rocblas-ubuntu}"
# MERGED=1: Tensile code objects carry BOTH architectures (Tensile_SEPARATE_ARCHITECTURES=OFF).  Needed on a box with
# cards of different architectures: HIP creates a loaded module on every device, and a gfx1200-only Tensile code object
# fails on the gfx1012 ("program ISA ... not compatible", then "no kernel image is available"), which Strata's next
# kernel error check picks up (FINDINGS.md, 2026-11-11).
MERGED="${MERGED:-1}"
# (Tensile allows lazy loading only with separate architectures, so the merged build loads its library at handle creation)
if [[ $MERGED == 1 ]]; then SEPARATE=OFF; LAZY=OFF; SUFFIX=-merged; else SEPARATE=ON; LAZY=ON; SUFFIX=; fi
PREFIX="${PREFIX:-$HOME/opt/rocblas-$VER-gfx1012-gfx1200$SUFFIX}"
BUILD="build$SUFFIX"
POOL=https://archive.ubuntu.com/ubuntu/pool/universe/r/rocblas

mkdir -p "$SRC"
cd "$SRC"
for f in "rocblas_$VER-$DEBREV.debian.tar.xz" "rocblas_$VER.orig.tar.gz" "rocblas_$VER.orig-tensile.tar.gz"; do
  [[ -f $f ]] || curl -sSfLO "$POOL/$f"
done

if [[ ! -f src/.patched ]]; then
  rm -rf src debian
  mkdir -p src/tensile
  tar xf "rocblas_$VER-$DEBREV.debian.tar.xz"
  tar xzf "rocblas_$VER.orig.tar.gz" -C src --strip-components=1
  tar xzf "rocblas_$VER.orig-tensile.tar.gz" -C src/tensile --strip-components=1
  while read -r p; do
    [[ -z $p || $p == \#* ]] && continue
    echo "patch: $p"
    patch -d src -p1 --forward --quiet < "debian/patches/$p"
  done < debian/patches/series
  touch src/.patched
fi

export CXX=hipcc CC=hipcc
export TENSILE_ROCM_ASSEMBLER_PATH=/usr/bin/clang++
export TENSILE_ROCM_OFFLOAD_BUNDLER_PATH=/usr/bin/clang-offload-bundler
export PATH="$PATH:/usr/lib/llvm-21/bin"

cmake -S src -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_SKIP_INSTALL_RPATH=ON \
  -DROCTX_PATH=/usr/lib/x86_64-linux-gnu \
  -DROCM_SYMLINK_LIBS=OFF \
  -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF \
  -DGPU_TARGETS="$TARGETS" \
  -DBUILD_WITH_HIPBLASLT=OFF \
  -DROCBLAS_TENSILE_LIBRARY_DIR="$PREFIX/lib/rocblas/5.1.0" \
  -DTensile_COMPILER=hipcc \
  -DTensile_TEST_LOCAL_PATH="../tensile" \
  -DTensile_ROOT="$SRC/src/tensile/Tensile" \
  -DTensile_LOGIC=asm_full \
  -DTensile_LIBRARY_FORMAT=msgpack \
  -DTensile_LAZY_LIBRARY_LOADING="$LAZY" \
  -DTensile_SEPARATE_ARCHITECTURES="$SEPARATE" \
  -DTensile_CPU_THREADS="$JOBS" \
  -DBUILD_CLIENTS_TESTS=OFF \
  -DBUILD_CLIENTS_BENCHMARKS=OFF \
  -DBUILD_CLIENTS_SAMPLES=OFF \
  -DBUILD_WITH_PIP=false \
  -DOptions=--ignore-asm-cap-cache \
  -DRUN_HEADER_TESTING=OFF \
  -DBUILD_OFFLOAD_COMPRESS=ON

nice -n 10 cmake --build "$BUILD" -j "$JOBS"
cmake --install "$BUILD"

{
  echo "rocBLAS $VER (Ubuntu $VER-$DEBREV sources + debian/patches), GPU_TARGETS=$TARGETS, Tensile_SEPARATE_ARCHITECTURES=$SEPARATE"
  echo "built $(date -Is) on $(hostname), hipcc $(hipcc --version 2>/dev/null | grep -m1 -o 'HIP version: [^ ]*')"
  echo "use: LD_LIBRARY_PATH=$PREFIX/lib"
} > "$PREFIX/BUILD-INFO.txt"
ls "$PREFIX/lib/rocblas/5.1.0/library" | grep -o 'gfx[0-9a-z]*' | sort -u | tr '\n' ' '; echo
