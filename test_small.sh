#!/usr/bin/env bash
# Quick check: VIMDF for 3 days of June 2020 into a separate test directory,
# then a summary of the year file it writes. Runs the same vimdf.run() the
# array job uses; only the date range and output directory are overridden.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO"

if [ ! -x "$PY" ]; then
    echo "error: no venv at $REPO/.venv; run ./setup.sh first" >&2
    exit 1
fi

"$PY" - <<'EOF'
import os
import time

import numpy as np
import xarray as xr

import vimdf

vimdf.START, vimdf.END = "2020-06-01", "2020-06-04"     # END exclusive: 3 days
vimdf.MONTHS = None
vimdf.OUT_DIR = os.path.join(os.path.dirname(vimdf.OUT_DIR), "test")
path = vimdf.year_path(2020)
if os.path.isfile(path):
    os.remove(path)                                     # run() skips existing years

t_start = time.perf_counter()
vimdf.run(0, 1)
elapsed = time.perf_counter() - t_start
print(f"\nrun took {elapsed:.0f} s for 3 days ({elapsed / 3:.0f} s/day)\n")

with xr.open_dataset(path) as ds:
    print(ds, "\n")
    assert ds.sizes["time"] == 3, f"expected 3 days, got {ds.sizes['time']}"
    assert list(ds["bounds"].values) == ["850-300", "700-300"], ds["bounds"].values

    for v in vimdf.VARIABLES:
        a = ds[v].values
        print(f"{v:8s} nan={int(np.isnan(a).sum()):7d}  min={np.nanmin(a):9.2f}  "
              f"max={np.nanmax(a):9.2f}  mean={np.nanmean(a):9.2f}")

    # The daily mean of hourly magnitudes can never be smaller than the
    # magnitude of the daily-mean vector (triangle inequality).
    of_mean = np.hypot(ds["vimdf_x"], ds["vimdf_y"])
    worst = float((ds["vimdf"] - of_mean).min())
    print(f"\nmin(mean|F| - |mean F|) = {worst:.4f}  (should be >= 0, up to float rounding)")
EOF
