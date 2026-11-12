"""Deoras et al. VIMDF (Eq. 1) from ERA5, read through wxread.

    python vimdf.py run <index> <count>    # the years belonging to shard <index> of <count>

Computes daily-mean VIMDF for every day from START to END and writes one file
per year, vimdf_<year>.nc, with dims (bounds, time, latitude, longitude).
Years are the unit of sharding, so each year file is written by exactly one
shard, and a shard skips any year whose file already exists.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from wxread import BBox, Calculation, Era5Rda, Request, plan, shard

# ---- configuration ----

BOX = BBox(lat_min=-35.0, lat_max=55.0, lon_min=15.0, lon_max=145.0)   # replace with the new box
START, END = "1940-01-01", "2024-01-01"          # END is exclusive: 1940 through 2023
MONTHS = None                                    # e.g. (6, 7, 8, 9) for JJAS only; None keeps all
LEVELS = (300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850)
BOUNDS = ((850.0, 300.0), (700.0, 300.0))        # (p_bottom, p_top) in hPa

# PENTAD_CENTERS = ((5, 15), (6, 15), (7, 15))     # used by `combine`
# HALF_WIDTH = 2                                   # +-2 days, so 5-day windows
OUT_DIR = os.path.join("/glade/derecho/scratch", os.environ.get("USER", ""),
                       "vimdf", "daily")
ENCODING = {"zlib": True, "complevel": 1, "dtype": "float32"}        # light compression on the year files

G0 = 9.80665                                     # m s-2
VARIABLES = ("vimdf_x", "vimdf_y", "vimdf")


def log(msg: str) -> None:
    print(f"[vimdf] {msg}", flush=True)


def bound_label(p_bottom: float, p_top: float) -> str:
    return f"{int(p_bottom)}-{int(p_top)}"


# def pentad_label(center: tuple[int, int]) -> str:
#     return f"{center[0]:02d}-{center[1]:02d}"


def year_path(year: int) -> str:
    return os.path.join(OUT_DIR, f"vimdf_{year}.nc")


def years() -> list[int]:
    """Years with at least one day in the request, from the calendar alone (no file access)."""
    days = pd.date_range(START, END, freq="1D", inclusive="left")
    if MONTHS is not None:
        days = days[days.month.isin(MONTHS)]
    return sorted(set(days.year))


def make_request(start: str, end: str) -> Request:
    calc = Calculation(name="vimdf", variables=("u", "v", "q", "t"), step="1D", levels=LEVELS)
    return Request(calc, start, end, BOX, months=MONTHS)


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


def day_dataset(day: date, daily: dict[str, xr.Dataset]) -> xr.Dataset:
    """One day's fields for every bound, with bounds and time dims added."""
    parts = [daily[bound_label(b, t)].expand_dims(bounds=[bound_label(b, t)])
             for b, t in BOUNDS]
    return xr.concat(parts, dim="bounds").expand_dims(time=[np.datetime64(day, "D")])


# ---- commands ----

def run(index: int, count: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    mine = shard(years(), index, count)
    log(f"shard {index} of {count}: years {mine}")
    source = Era5Rda()

    for year in mine:
        out_path = year_path(year)
        if os.path.isfile(out_path):
            log(f"{out_path} exists; skipping {year}")
            continue

        # ISO date strings compare correctly as text, so max/min clip the year to START/END.
        start = max(START, f"{year}-01-01")
        end = min(END, f"{year + 1}-01-01")
        steps = plan(make_request(start, end), source)
        log(f"{year}: {len(steps)} days")

        days = []
        n_missing = 0
        t_start = time.perf_counter()
        for i, step in enumerate(steps, 1):
            if step.missing:
                log(f"{step.label}: missing {step.missing}, skipped")
                n_missing += 1
                continue
            ds = step.open(chunks=None).load()
            days.append(day_dataset(step.t0.date(), daily_vimdf(ds)))
            ds.close()
            if i % 25 == 0:
                rate = (time.perf_counter() - t_start) / i
                log(f"{step.label}: {i}/{len(steps)} days, {rate:.1f} s/day, "
                    f"~{rate * (len(steps) - i) / 60:.0f} min left in {year}")

        if not days:
            raise RuntimeError(f"no days were read for {year}")
        out = xr.concat(days, dim="time")
        for v in VARIABLES:
            out[v].attrs["units"] = "kg m-1 s-1"
        out.attrs["source"] = "Deoras et al., Eq. 1: (1/g) int u (qs - q) dp; ERA5 via wxread"
        out.attrs["description"] = "daily mean of hourly VIMDF"
        out.attrs["n_days_missing"] = n_missing

        tmp_path = out_path + ".tmp"
        out.to_netcdf(tmp_path, encoding={v: ENCODING for v in VARIABLES})
        os.replace(tmp_path, out_path)
        log(f"wrote {out_path}: {len(days)} days ({n_missing} days skipped for missing files)")


# def combine() -> None:
#     all_years = years()
#     missing = [y for y in all_years if not os.path.isfile(year_path(y))]
#     if missing:
#         raise FileNotFoundError(f"missing year files for {missing}")
#
#     sums: dict[tuple[int, int], xr.Dataset] = {}
#     counts: dict[tuple[int, int], int] = {}
#     for year in all_years:
#         with xr.open_dataset(year_path(year)) as ds:
#             times = pd.DatetimeIndex(ds["time"].values)
#             for center in PENTAD_CENTERS:
#                 offset_days = (times - pd.Timestamp(year, *center)).days
#                 idx = np.where(np.abs(offset_days) <= HALF_WIDTH)[0]
#                 if len(idx) == 0:
#                     continue
#                 s = ds[list(VARIABLES)].isel(time=idx).sum("time").load()
#                 s = s.map(lambda v: v.astype("float64"))
#                 sums[center] = sums[center] + s if center in sums else s
#                 counts[center] = counts.get(center, 0) + len(idx)
#
#     found = [c for c in PENTAD_CENTERS if c in sums]
#     if not found:
#         raise ValueError("no year file contains a pentad day; check MONTHS against PENTAD_CENTERS")
#     means = []
#     for center in found:
#         m = (sums[center] / counts[center]).expand_dims(pentad=[pentad_label(center)])
#         m["n_days"] = ("pentad", [counts[center]])
#         means.append(m)
#     out = xr.concat(means, dim="pentad")
#     for v in VARIABLES:
#         out[v].attrs["units"] = "kg m-1 s-1"
#     out.attrs["description"] = (f"pentad-mean VIMDF, {2 * HALF_WIDTH + 1}-day windows "
#                                 f"pooled over {all_years[0]}-{all_years[-1]}")
#
#     out_path = os.path.join(OUT_DIR, "vimdf_pentads.nc")
#     out.to_netcdf(out_path)
#     log(f"wrote {out_path}")
#     log(f"days per pentad: {dict(zip(out['pentad'].values, out['n_days'].values))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("index", type=int)
    r.add_argument("count", type=int)
    args = parser.parse_args()
    run(args.index, args.count)

if __name__ == "__main__":
    main()
