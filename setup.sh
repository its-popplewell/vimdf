#!/usr/bin/env bash
# One-time setup for the VIMDF pentad run. Run from anywhere:  ./setup.sh
#
# Creates a project-local Python venv with what vimdf.py and wxread.py need,
# the logs/ directory, and the output directory named by OUT_DIR in vimdf.py.
# Override the base interpreter with PYTHON=/path/to/python3 ./setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$REPO/.venv"
ERA5_ROOT="/glade/campaign/collections/rda/data/d633000"

cd "$REPO"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: '$PYTHON' not found on PATH" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: need Python 3.10 or newer; '$PYTHON' is $("$PYTHON" --version 2>&1)" >&2
    echo "on NCAR systems, try:  module load conda && conda activate npl  then rerun" >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "creating venv at $VENV"
    "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet numpy pandas xarray h5netcdf netCDF4

mkdir -p "$REPO/logs"
OUT_DIR="$("$VENV/bin/python" -c 'import os, vimdf; os.makedirs(vimdf.OUT_DIR, exist_ok=True); print(vimdf.OUT_DIR)')"

if [ ! -d "$ERA5_ROOT" ]; then
    echo "warning: ERA5 archive not found at $ERA5_ROOT; the run reads from there" >&2
fi

NSHARDS=$(( $(sed -nE 's/^#PBS -J 0-([0-9]+).*/\1/p' "$REPO/jobs/vimdf_shards.pbs") + 1 ))

echo
echo "setup done"
echo "  venv:    $VENV"
echo "  output:  $OUT_DIR"
echo "  shards:  $NSHARDS"
echo
echo "set BOX in vimdf.py if you haven't, then submit from $REPO with:"
echo "  qsub -A <your-project-code> -o \"$REPO/logs/\" jobs/vimdf_shards.pbs"
echo "when every shard has finished:"
echo "  ./finish.sh"
