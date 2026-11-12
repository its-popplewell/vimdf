"""Deoras et al. VIMDF (Eq. 1) pentad means from ERA5, read through wxread.

    python vimdf.py run <index> <count>    # one shard of the pentad days
    python vimdf.py combine                # sum the shard files into pentad means

Each shard writes running sums and day counts rather than means, so shards
holding different numbers of days combine correctly.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from datetime import date

import numpy as np
import xarray as xr

from wxread import BBox, Calculation, Era5Rda, Request, plan, shard

# ---- configuration ----

BOX = BBox(lat_min=-35.0, lat_max=55.0, lon_min=15.0, lon_max=145.0)   # replace with the new box
START, END = "1940-05-01", "2023-08-01"          # END is exclusive
LEVELS = (300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850)
BOUNDS = ((850.0, 300.0), (700.0, 300.0))        # (p_bottom, p_top) in hPa
PENTAD_CENTERS = ((5, 15), (6, 15), (7, 15))     # (month, day)
HALF_WIDTH = 2                                   # +-2 days, so 5-day windows
OUT_DIR = os.path.join("/glade/derecho/scratch", os.environ.get("USER", ""),
                       "vimdf", "pentads")

G0 = 9.80665                                     # m s-2


def log(msg: str) -> None:
    print(f"[vimdf] {msg}", flush=True)


def bound_label(p_bottom: float, p_top: float) -> str:
    return f"{int(p_bottom)}-{int(p_top)}"


def pentad_label(center: tuple[int, int]) -> str:
    return f"{center[0]:02d}-{center[1]:02d}"


def pentads_containing(d: date) -> list[tuple[int, int]]:
    return [c for c in PENTAD_CENTERS
            if abs((d - date(d.year, *c)).days) <= HALF_WIDTH]


# ---- Eq. 1 ----

def saturation_q(t, p_pa):
    """Saturation specific humidity, Bolton (1980) es over water. t in K, p in Pa."""
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p_pa - 0.378 * es)


def daily_vimdf(ds: xr.Dataset) -> dict[str, xr.Dataset]:
    """Eq. 1 on each hour, then the day mean, for every pair in BOUNDS.

    VIMDF is nonlinear in its inputs, so integrate first and average second.
    The moisture deficit is computed once and shared by both bounds.
    """
    ds = ds.map(lambda v: v.astype("float32")).sortby("level")   # ascending p: top -> bottom
    p_pa = ds["level"] * 100.0
    deficit = saturation_q(ds["t"], p_pa) - ds["q"]

    out = {}
    for p_bottom, p_top in BOUNDS:
        layer = {"level": slice(p_top, p_bottom)}
        d = deficit.sel(layer)
        fx = (ds["u"].sel(layer) * d).integrate("level") * 100.0 / G0   # *100: hPa -> Pa
        fy = (ds["v"].sel(layer) * d).integrate("level") * 100.0 / G0
        hourly = xr.Dataset({"vimdf_x": fx, "vimdf_y": fy, "vimdf": np.hypot(fx, fy)})
        out[bound_label(p_bottom, p_top)] = hourly.mean("time")
    return out


# ---- per-shard sums ----

class ShardSums:
    """Running sums and day counts per (bounds, pentad)."""

    def __init__(self):
        self.sums: dict[tuple[str, tuple[int, int]], xr.Dataset] = {}
        self.counts: dict[tuple[str, tuple[int, int]], int] = {}
        self.template: xr.Dataset | None = None

    def add(self, d: date, daily: dict[str, xr.Dataset]) -> None:
        for center in pentads_containing(d):
            for label, field in daily.items():
                field = field.map(lambda v: v.astype("float64"))
                key = (label, center)
                if key in self.sums:
                    self.sums[key] = self.sums[key] + field
                else:
                    self.sums[key] = field
                self.counts[key] = self.counts.get(key, 0) + 1
                if self.template is None:
                    self.template = xr.zeros_like(field)

    def to_dataset(self) -> xr.Dataset:
        per_bound = []
        for p_bottom, p_top in BOUNDS:
            label = bound_label(p_bottom, p_top)
            per_pentad = []
            for center in PENTAD_CENTERS:
                key = (label, center)
                s = self.sums.get(key, self.template)        # zeros if this shard had none
                s = s.rename({v: f"sum_{v}" for v in s.data_vars})
                s["n_days"] = self.counts.get(key, 0)
                per_pentad.append(s.expand_dims(pentad=[pentad_label(center)]))
            per_bound.append(xr.concat(per_pentad, dim="pentad")
                         .expand_dims(bounds=[label]))
        return xr.concat(per_bound, dim="bounds")


# ---- commands ----

def run(index: int, count: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"shard_{index:03d}_of_{count:03d}.nc")
    if os.path.isfile(out_path):
        log(f"{out_path} exists; nothing to do")
        return

    calc = Calculation(name="vimdf", variables=("u", "v", "q", "t"), step="1D", levels=LEVELS)
    months = tuple(sorted({m for m, _ in PENTAD_CENTERS}))
    request = Request(calc, START, END, BOX, months=months)
    steps = [s for s in plan(request, Era5Rda()) if pentads_containing(s.t0.date())]
    mine = shard(steps, index, count)
    log(f"shard {index} of {count}: {len(mine)} of {len(steps)} pentad days")

    sums = ShardSums()
    n_missing = 0
    t_start = time.perf_counter()
    for i, step in enumerate(mine, 1):
        if step.missing:
            log(f"{step.label}: missing {step.missing}, skipped")
            n_missing += 1
            continue
        ds = step.open(chunks=None).load()
        sums.add(step.t0.date(), daily_vimdf(ds))
        ds.close()
        if i % 10 == 0:
            rate = (time.perf_counter() - t_start) / i
            log(f"{step.label}: {i}/{len(mine)} days, {rate:.1f} s/day, "
                f"~{rate * (len(mine) - i) / 60:.0f} min left")

    if sums.template is None:
        raise RuntimeError("no days were read in this shard")
    tmp_path = out_path + ".tmp"
    sums.to_dataset().to_netcdf(tmp_path)
    os.replace(tmp_path, out_path)
    log(f"wrote {out_path} ({n_missing} days skipped for missing files)")


def combine() -> None:
    files = sorted(glob.glob(os.path.join(OUT_DIR, "shard_*.nc")))
    if not files:
        raise FileNotFoundError(f"no shard files in {OUT_DIR}")
    total = None
    for path in files:
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        total = ds if total is None else total + ds

    means = xr.Dataset({v: total[f"sum_{v}"] / total["n_days"]
                        for v in ("vimdf_x", "vimdf_y", "vimdf")})
    means["n_days"] = total["n_days"]
    for v in ("vimdf_x", "vimdf_y", "vimdf"):
        means[v].attrs["units"] = "kg m-1 s-1"
    means.attrs["source"] = "Deoras et al., Eq. 1: (1/g) int u (qs - q) dp; ERA5 via wxread"
    means.attrs["description"] = (f"pentad-mean VIMDF, {2 * HALF_WIDTH + 1}-day windows "
                                  f"pooled over {START} to {END}")

    out_path = os.path.join(OUT_DIR, "vimdf_pentads.nc")
    means.to_netcdf(out_path)
    log(f"combined {len(files)} shards into {out_path}")
    log(f"days per (bounds, pentad):\n{means['n_days'].to_pandas()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("index", type=int)
    r.add_argument("count", type=int)
    sub.add_parser("combine")
    args = parser.parse_args()
    if args.command == "run":
        run(args.index, args.count)
    else:
        combine()


if __name__ == "__main__":
    main()
