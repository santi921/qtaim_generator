#!/usr/bin/env bash
# Build Chargemol 09_26_2017 (Manz and Gabaldon Limas, ddec.sourceforge.net) as the
# DDEC6 validation reference. Never used by the pipeline.
#
# Builds four binaries from the source in <chargemol dir>, all with
# chargemol_build_fix.patch: modulo(k, gridspacing) is evaluated even when
# gridspacing = 0 (analytic pairs) because Fortran does not short-circuit .and.;
# gfortran -O0 then dies with SIGFPE. max(gridspacing, 1) changes no result, since
# the guard (gridspacing > 1) is already false there.
#   chargemol_serial, chargemol_omp                  stock output (f13.6 charges)
#   chargemol_serial_precise, chargemol_omp_precise  + chargemol_precision.patch,
#       which only widens the per-atom output formats (charges, multipoles, spin
#       moments, radial moments, tail fits, bond orders) to es25.16 so the port can
#       be compared beyond 1e-6.
#   chargemol_serial_dump  precise + chargemol_dump.patch, which writes the pair data,
#       the grids after Step 7 and the final grids of the density module to
#       dump_pairs.bin, dump_step7.bin and dump_final.bin (stream access, column-major)
#       for checking the density-grid port kernel by kernel.
# The Chargemol license allows distributing patch files for build-time changes; the
# source itself is not redistributed here.
#
# usage: build_chargemol.sh <chargemol_09_26_2017 dir> <out dir>
# FFLAGS defaults to empty, as in Chargemol's compile_serial.txt (gfortran -O0).
set -euo pipefail

SRC="$1/chargemol_FORTRAN_09_26_2017/sourcecode_linux"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
FFLAGS="${FFLAGS:-}"

mapfile -t FILES < <(grep -o 'module_[A-Za-z0-9_]*\.f08' "$SRC/compile_serial.txt")
FILES+=(chargemol.f08)

build() {
    local name=$1 precise=$2 omp=$3 tmp
    tmp="$(mktemp -d)"
    cp "$SRC"/*.f08 "$tmp"
    (cd "$tmp" && patch -p1 -s < "$HERE/chargemol_build_fix.patch")
    if [ "$precise" -ge 1 ]; then
        (cd "$tmp" && patch -p1 -s < "$HERE/chargemol_precision.patch")
    fi
    if [ "$precise" = 2 ]; then
        (cd "$tmp" && patch -p1 -s < "$HERE/chargemol_dump.patch")
    fi
    # shellcheck disable=SC2086
    (cd "$tmp" && gfortran $FFLAGS $omp -static-libgfortran -o "$OUT/$name" "${FILES[@]}")
    rm -rf "$tmp"
    echo "built $OUT/$name"
}

build chargemol_serial 0 ""
build chargemol_omp 0 "-fopenmp"
build chargemol_serial_precise 1 ""
build chargemol_omp_precise 1 "-fopenmp"
build chargemol_serial_dump 2 ""
