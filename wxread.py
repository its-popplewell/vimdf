"""
wxread — a small reusable reading engine for gridded reanalysis / model output.

Concepts
--------
Calculation  What variables and levels a computation needs, and the smallest
             self-contained timestep it operates on.
Request      A Calculation + a date range + a geographic bbox.
Source       Knows how a dataset is laid out on disk and what its variables are
             called natively. Era5Rda now; Cesm2 slots in beside it later.
Step         One label plus an `open()` that returns an xarray Dataset holding
             exactly that timestep, with canonical variable names.

    steps = plan(request, Era5Rda())
    for step in steps:
        ds = step.open()          # first disk access happens here
        result = my_calc(ds)

Steps are plain frozen dataclasses, so they pickle cheaply. Any of these work
without changing the engine:

    for s in steps: ...                                  # serial
    with Pool(8) as p: p.map(run, steps)                 # one node
    for s in shard(steps, rank, nranks): ...             # PBS array job
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Protocol, Sequence

import pandas as pd
import xarray as xr

# --------------------------------------------------------------------------
# Request description
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """Geographic box. Longitudes in -180..180 OR 0..360; sources normalize."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class Calculation:
    """What a computation needs, independent of any date range or region.

    variables : canonical names ("u", "v", "q", "sp", ...) — see Source maps.
    step      : pandas offset alias for the *stride* between timesteps
                ("1h", "6h", "1D", "MS").
    levels    : pressure levels in hPa, or None for no level selection.
    window    : how long each timestep spans. Defaults to `step`, which
                partitions the range into contiguous non-overlapping windows.
                Set it shorter than `step` for windows with gaps between them,
                or longer for overlapping / running windows.
    offset    : shifts each window's start off the natural boundary.

    An overnight window — 18Z to 06Z, one per day — is:

        step="1D", window="12h", offset="18h"

    Windows are indexed by their start, so a request's date range is the range
    of window *starts*; a trailing window may extend past `end`.
    """

    name: str
    variables: tuple[str, ...]
    step: str
    levels: tuple[float, ...] | None = None
    window: str | None = None
    offset: str | None = None


@dataclass(frozen=True)
class Request:
    """A calculation over a date range and a region.

    months : keep only windows whose start falls in these months, or None for
             all of them. Membership rather than a range, so a season that
             wraps the year end needs no special handling: November through
             March is (11, 12, 1, 2, 3).

             The test is applied to the window start *after* `offset`, which
             is the value in Step.label. So a Jones nighttime window starting
             23Z on 31 March is kept while the one starting 23Z on 1 April is
             dropped, matching a 1 Nov - 31 Mar season. That kept window still
             reads 1 April's file for its tail hours; the filter selects
             labels, `files()` covers whatever the window spans, and the two
             legitimately disagree at a month boundary.
    """

    calculation: Calculation
    start: str  # inclusive, e.g. "1990-01-01"
    end: str  # exclusive, e.g. "1990-02-01"
    bbox: BBox
    months: tuple[int, ...] | None = None


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


class Source(Protocol):
    """A dataset's on-disk layout and naming conventions."""

    def files(self, variable: str, t0: datetime, t1: datetime) -> tuple[str, ...]:
        """Paths holding `variable` over [t0, t1). May be empty."""

    def native_name(self, variable: str) -> str:
        """In-file variable name for a canonical name."""

    def subset(
        self,
        ds: xr.Dataset,
        bbox: BBox,
        levels: tuple[float, ...] | None,
        t0: datetime,
        t1: datetime,
    ) -> xr.Dataset:
        """Slice to the requested region / levels / time window."""

    def timeline(self, start: str, end: str, step: str) -> pd.DatetimeIndex:
        """Step boundaries. Overridden for non-standard calendars."""

    def chunks(self, variable: str, chunks: dict) -> dict:
        """Requested chunks, minus any dims this variable doesn't have."""


# ERA5 on NCAR's GLADE, from the RDA collection.
#
# canonical -> (stream, parameter code, grid tag)
ERA5_VARS: dict[str, tuple[str, str, str]] = {
    # pressure-level analysis, one file per day
    "u": ("an.pl", "128_131_u", "ll025uv"),
    "v": ("an.pl", "128_132_v", "ll025uv"),
    "w": ("an.pl", "128_135_w", "ll025sc"),
    "t": ("an.pl", "128_130_t", "ll025sc"),
    "q": ("an.pl", "128_133_q", "ll025sc"),
    "z": ("an.pl", "128_129_z", "ll025sc"),
    # surface analysis, one file per month
    "sp": ("an.sfc", "128_134_sp", "ll025sc"),
    "tcwv": ("an.sfc", "128_137_tcwv", "ll025sc"),
    "t2m": ("an.sfc", "128_167_2t", "ll025sc"),
}

ERA5_NATIVE: dict[str, str] = {
    "u": "U",
    "v": "V",
    "w": "W",
    "t": "T",
    "q": "Q",
    "z": "Z",
    "sp": "SP",
    "tcwv": "TCWV",
    "t2m": "VAR_2T",
}


@dataclass(frozen=True)
class Era5Rda:
    # RDA renumbered its datasets; the old id was ds633.0.
    root: str = "/glade/campaign/collections/rda/data/d633000"

    def native_name(self, variable: str) -> str:
        return ERA5_NATIVE[variable]

    def timeline(self, start: str, end: str, step: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq=step, inclusive="left")

    def chunks(self, variable: str, chunks: dict) -> dict:
        stream, _, _ = ERA5_VARS[variable]
        if stream != "an.pl":  # surface streams have no level dim
            return {k: v for k, v in chunks.items() if k != "level"}
        return chunks

    def files(self, variable: str, t0: datetime, t1: datetime) -> tuple[str, ...]:
        stream, code, grid = ERA5_VARS[variable]
        out: list[str] = []
        if stream == "an.pl":
            for day in _days(t0, t1):
                pat = (
                    f"{self.root}/e5.oper.{stream}/{day:%Y%m}/"
                    f"e5.oper.{stream}.{code}.{grid}.{day:%Y%m%d}*.nc"
                )
                out.extend(sorted(glob.glob(pat)))
        else:  # monthly surface files
            for month in _months(t0, t1):
                pat = (
                    f"{self.root}/e5.oper.{stream}/{month:%Y%m}/"
                    f"e5.oper.{stream}.{code}.{grid}.{month:%Y%m}01*.nc"
                )
                out.extend(sorted(glob.glob(pat)))
        return tuple(dict.fromkeys(out))

    def subset(
        self,
        ds: xr.Dataset,
        bbox: BBox,
        levels: tuple[float, ...] | None,
        t0: datetime,
        t1: datetime,
    ) -> xr.Dataset:
        # ERA5 latitude runs 90 -> -90, longitude 0 -> 360.
        ds = ds.sel(latitude=slice(bbox.lat_max, bbox.lat_min))
        ds = _sel_lon_0_360(ds, "longitude", bbox.lon_min, bbox.lon_max)
        if levels is not None and "level" in ds.dims:
            ds = ds.sel(level=list(levels))
        return ds.sel(time=slice(t0, t1 - timedelta(microseconds=1)))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Era5Local(Era5Rda):
    """A regional extraction produced by extract_era5.jl.

    One file per day holding every variable, so `files` returns the same path
    for each. Inherits `subset`, `chunks` and `timeline` from Era5Rda because
    the extraction preserves ERA5's conventions: latitude descending,
    longitude on 0..360, level ascending in hPa, time as a real time coord.
    Only `native_name` differs, since the extracted files use the canonical
    names directly rather than U/V/Q/T/SP.

    If the extracted files hold a whole day in one chunk per variable, pass
    chunks=None to Step.open — sub-chunking a single chunk gains nothing.
    """

    root: str = os.path.join(
        "/glade/derecho/scratch", os.environ.get("USER", ""), "era5_regional"
    )

    def native_name(self, variable: str) -> str:
        return variable

    def files(self, variable: str, t0: datetime, t1: datetime) -> tuple[str, ...]:
        # Every variable lives in the same daily file.
        paths = [
            os.path.join(self.root, f"{day:%Y%m}", f"era5_{day:%Y%m%d}.nc")
            for day in _days(t0, t1)
        ]
        return tuple(p for p in paths if os.path.isfile(p))


# Chunks requested at open time. {"time": 1} matches both the RDA collection
# (one timestep per chunk) and the regional extraction, and leaves `level`
# unchunked so all levels come as one piece — which is how both store them.
# Verify against a real file with:
#   ds["U"].encoding["chunksizes"]      # or probe_rda.py section 3
DEFAULT_CHUNKS: dict[str, int] = {"time": 1}


@dataclass(frozen=True)
class Step:
    """One self-contained timestep. Holds no open handles and no dask graph."""

    label: str
    # datetime for reanalysis, cftime for model calendars -- whatever the
    # source's timeline produced, carried through unconverted.
    t0: object
    t1: object
    paths: tuple[tuple[str, tuple[str, ...]], ...]  # (canonical var, files)
    bbox: BBox
    levels: tuple[float, ...] | None
    source: Source

    def open(self, chunks: dict | None = DEFAULT_CHUNKS) -> xr.Dataset:
        """Read this step as dask-backed arrays. Pass an explicit chunks dict
        to override, or chunks=None to read eagerly into numpy."""
        requested = dict(chunks) if chunks is not None else None
        parts = []
        for var, files in self.paths:
            native = self.source.native_name(var)
            # Chunks go in at open time, not via a later .chunk() call: one
            # chunk per file would make dask load a whole file to serve a
            # 12-hour slice out of it.
            ck = None if requested is None else self.source.chunks(var, requested)
            if len(files) == 1:
                ds = xr.open_dataset(
                    files[0], engine="h5netcdf", chunks=ck, decode_timedelta=False
                )
            else:
                ds = xr.open_mfdataset(
                    list(files),
                    engine="h5netcdf",
                    combine="by_coords",
                    chunks=ck if ck is not None else {},
                    decode_timedelta=False,
                )
            ds = ds[[native]].rename({native: var})
            parts.append(
                self.source.subset(ds, self.bbox, self.levels, self.t0, self.t1)
            )
        return xr.merge(parts, join="exact", combine_attrs="drop_conflicts")

    @property
    def missing(self) -> tuple[str, ...]:
        """Canonical variables with no file on disk for this step."""
        return tuple(v for v, f in self.paths if not f)


def _duration(alias: str):
    """A window length usable with any calendar's timestamps.

    A fixed-length Timedelta where the alias has one ("10h", "1D"), since
    that adds cleanly to a cftime object as well as to a pandas Timestamp.
    Calendar-relative aliases ("MS", "YS") have no fixed length and fall back
    to a pandas offset, which only works on pandas timestamps -- a monthly
    window on a noleap calendar would need cftime arithmetic instead, and
    raising there is better than being subtly wrong.
    """
    try:
        return pd.Timedelta(alias)
    except ValueError:
        return pd.tseries.frequencies.to_offset(alias)


def _plain(t):
    """A pandas Timestamp as a plain datetime; anything else untouched.

    ERA5 timelines come from pandas, and the steps have always carried
    datetimes. cftime objects (noleap model calendars) have no
    `to_pydatetime` and must stay as they are, since converting them would
    reintroduce the leap days the calendar does not have.
    """
    return t.to_pydatetime() if hasattr(t, "to_pydatetime") else t


def plan(request: Request, source: Source) -> list[Step]:
    """Expand a request into an ordered list of independent steps."""
    calc = request.calculation
    length = _duration(calc.window or calc.step)
    shift = pd.Timedelta(calc.offset) if calc.offset else pd.Timedelta(0)
    steps: list[Step] = []
    for edge in source.timeline(request.start, request.end, calc.step):
        t0 = _plain(edge + shift)
        t1 = _plain(edge + shift + length)
        # Filtered on the shifted start, i.e. on what appears in the label.
        if request.months is not None and t0.month not in request.months:
            continue
        steps.append(
            Step(
                # strftime, not a format spec: cftime objects implement the
                # former but not __format__.
                label=f"{calc.name}/{t0.strftime('%Y%m%dT%H%M')}",
                t0=t0,
                t1=t1,
                paths=tuple((v, source.files(v, t0, t1)) for v in calc.variables),
                bbox=request.bbox,
                levels=calc.levels,
                source=source,
            )
        )
    return steps


def shard(steps: Sequence[Step], index: int, count: int) -> list[Step]:
    """Steps belonging to worker `index` of `count`. Strided, not blocked, so
    slow regions of the timeline spread evenly across workers."""
    return list(steps[index::count])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _days(t0, t1) -> Iterator:
    """Midnight of each day touched by [t0, t1).

    Truncating t0 to midnight is what makes an overnight window pull two
    files: 23Z on the 30th truncates to the 30th, and the loop continues
    while day < t1.

    `replace` rather than constructing a datetime, so the timestamps stay in
    whatever calendar the source uses. Building a datetime here would put
    29 February back into a noleap model calendar.
    """
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < t1:
        yield day
        day += timedelta(days=1)


def _months(t0, t1) -> Iterator:
    """The first of each month touched by [t0, t1).

    Stepping 32 days then snapping back to the first lands in the next month
    whatever its length, and avoids month arithmetic special cases.
    """
    month = t0.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while month < t1:
        yield month
        month = (month + timedelta(days=32)).replace(day=1)


def _sel_lon_0_360(ds: xr.Dataset, name: str, lon_min: float, lon_max: float):
    """Select a longitude band on a 0..360 ascending axis, accepting negative
    inputs and handling a band that crosses the prime meridian."""
    a, b = lon_min % 360, lon_max % 360
    if a <= b:
        return ds.sel({name: slice(a, b)})
    left = ds.sel({name: slice(a, 360)})
    right = ds.sel({name: slice(0, b)})
    return xr.concat([left, right], dim=name)
