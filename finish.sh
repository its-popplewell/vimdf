#!/usr/bin/env bash
# After the shard array finishes: check every shard wrote its file, report
# days skipped for missing ERA5 files, then combine into pentad means.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO"

if [ ! -x "$PY" ]; then
    echo "error: no venv at $REPO/.venv; run ./setup.sh first" >&2
    exit 1
fi

# The shard count comes from the job script's array range, so the two can't disagree.
NSHARDS=$(( $(sed -nE 's/^#PBS -J 0-([0-9]+).*/\1/p' "$REPO/jobs/vimdf_shards.pbs") + 1 ))
OUT_DIR="$("$PY" -c 'import vimdf; print(vimdf.OUT_DIR)')"

missing=()
for ((i = 0; i < NSHARDS; i++)); do
    f=$(printf "%s/shard_%03d_of_%03d.nc" "$OUT_DIR" "$i" "$NSHARDS")
    [ -f "$f" ] || missing+=("$i")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "missing ${#missing[@]} of $NSHARDS shards: ${missing[*]}" >&2
    echo "resubmit the array; shards that already finished exit immediately:" >&2
    echo "  qsub -A <your-project-code> -o \"$REPO/logs/\" jobs/vimdf_shards.pbs" >&2
    exit 1
fi

if ls "$OUT_DIR"/*.tmp >/dev/null 2>&1; then
    echo "warning: leftover .tmp files in $OUT_DIR from an interrupted shard" >&2
fi

echo "all $NSHARDS shards present"
if ls "$REPO"/logs/* >/dev/null 2>&1; then
    echo "missing-file skips reported by shards:"
    grep -h "days skipped for missing files" "$REPO"/logs/* || echo "  (none found in logs)"
fi

"$PY" vimdf.py combine
