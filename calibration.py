#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible comparison of Mandria calibration methods using original CSV/XLS files.

Reads either the two extracted input directories or a ZIP archive directly.
Gas data: 30-second means; entire local calendar days are assigned to splits.
PM data: daily averages only, paired with daily reference data from the original XLS.

Windows example:
    python calibration_paris_english.py --check
    python calibration_paris_english.py

Custom data directory or ZIP archive:
    python calibration_paris_english.py --data-root "C:\\...\\Background_Data_Paris"
    python calibration_paris_english.py --zip "C:\\...\\zip.zip"

Important: 30-second observations are autocorrelated; independent test units
are calendar days. Two PM test days cannot provide robust independent validation.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# USER-CONFIGURABLE DEFAULTS
# -----------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path(r"C:\Users\xy-84\Desktop\Master\Thesis\Background_Data_Paris")
TZ = "Europe/Rome"                 # Measurement site; same daylight-saving time as Paris
SAMPLE_SECONDS = 30
MIN_RAW_COVERAGE = 0.95           # Raw sensor: >=95% of 30-second intervals per day
MIN_REF_COVERAGE = 0.80           # Gas reference: >=80% valid values per day
MIN_PAIRED_COVERAGE = 0.75        # Paired gas observations per day
DATE_FROM = date(2026, 9, 24)
DATE_TO = date(2026, 10, 5)       # inclusive (partial days excluded by coverage checks)
GAS_FEATURES_P = ["CO_raw", "NO2_raw", "O3_raw", "T", "RH", "p"]
GAS_FEATURES_NO_P = [c for c in GAS_FEATURES_P if c != "p"]
PRIMARY_GAS = {"CO": "CO_raw", "NO2": "NO2_raw", "O3": "O3_raw"}
RAW_MAP = {
    "mics_reducing_ohm": "CO_raw",      # physical proxy sensor
    "mics_oxidising_ohm": "NO2_raw",     # cross-sensitive, NOT selective
    "mq131_raw": "O3_raw",              # cross-sensitive, NOT selective
    "temp_c": "T", "humidity_percent": "RH", "pressure_hpa": "p",
    "pm2_5_ug_m3": "PM2_5_raw", "pm10_ug_m3": "PM10_raw",
}
CANONICAL_NAMES = {
    "gas_drone": "gas_drone", "ground_basic": "ground_basic",
    "ground_screen": "ground_screen", "opc_drone": "opc_drone",
    "thermo_reference": "thermo_reference", "pm_reference": "pm_reference",
}


@dataclass(frozen=True)
class Source:
    member: str                   # relative archive member name or absolute file path
    archive: Path | None = None

    @property
    def basename(self) -> str:
        return Path(self.member).name

    @contextlib.contextmanager
    def open(self):
        if self.archive is not None:
            with zipfile.ZipFile(self.archive) as z:
                with z.open(self.member) as f:
                    yield f
        else:
            with Path(self.member).open("rb") as f:
                yield f

    def label(self) -> str:
        return f"{self.archive}::{self.member}" if self.archive else self.member


def classify(name: str) -> str | None:
    s = Path(name).name.lower()
    if s.endswith(".csv"):
        for key in ("ground_basic", "ground_screen", "gas_drone", "opc_drone"):
            if key in s:
                return key
        if "thermo_reference" in s:
            return "thermo_reference"
    if s.endswith((".xls", ".xlsx")) and "pm10" in s and "pm2" in s:
        return "pm_reference"
    return None


def find_inputs(root: Path, zip_file: Path | None) -> dict[str, Source]:
    found: dict[str, Source] = {}
    # An explicitly provided ZIP takes precedence.
    archives = []
    if zip_file:
        if not zip_file.is_file():
            raise FileNotFoundError(f"ZIP archive not found: {zip_file}")
        archives = [zip_file]
    else:
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {root}")
        for p in (
            sorted((root / "Raw data calibration Mandria").glob("*"))
            + sorted((root / "Calibrated comparison data Mandria").glob("*"))
        ):
            if p.is_file() and p.suffix.lower() in (".csv", ".xls", ".xlsx"):
                typ = classify(p.name)
                if typ and typ not in found:
                    found[typ] = Source(str(p))
        archives = []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                typ = classify(name)
                if typ and (zip_file is not None or typ not in found):
                    found[typ] = Source(name, archive)
    needed = set(CANONICAL_NAMES)
    missing = sorted(needed - set(found))
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing)
                                + f". Searched directory: {root}")
    return found


def csv_header(src: Source) -> list[str]:
    with src.open() as f:
        return pd.read_csv(f, nrows=0, encoding="utf-8-sig").columns.tolist()


def read_sensor_binned(src: Source, chunksize: int = 170_000) -> pd.DataFrame:
    """Read a 1-Hz CSV in chunks and compute exact UTC 30-second means."""
    header = csv_header(src)
    mandatory = ["gps_date", "gps_time_utc"]
    if any(c not in header for c in mandatory):
        raise ValueError(f"Missing timestamp columns: {src.label()}")
    feature_cols = [c for c in RAW_MAP if c in header]
    if not feature_cols:
        raise ValueError(f"No sensor columns found: {src.label()}")
    read_cols = mandatory + feature_cols
    sum_parts, count_parts = [], []
    n_total, n_good = 0, 0
    with src.open() as f:
        for chunk in pd.read_csv(f, usecols=read_cols, chunksize=chunksize,
                                 encoding="utf-8-sig", low_memory=False):
            n_total += len(chunk)
            times = pd.to_datetime(chunk["gps_date"].astype("string") + " "
                                   + chunk["gps_time_utc"].astype("string"),
                                   errors="coerce", format="mixed", utc=True)
            valid = times.notna() & (times.dt.year == 2026)
            valid &= (times >= pd.Timestamp(DATE_FROM, tz="UTC") - pd.Timedelta(days=2))
            valid &= (times < pd.Timestamp(DATE_TO, tz="UTC") + pd.Timedelta(days=2))
            if not valid.any():
                continue
            n_good += int(valid.sum())
            clean = pd.DataFrame(index=pd.DatetimeIndex(times[valid], name="time_utc"))
            for old in feature_cols:
                clean[RAW_MAP[old]] = pd.to_numeric(chunk.loc[valid, old].to_numpy(), errors="coerce")
            # Mask physically implausible values and obvious ADC saturation.
            if "T" in clean:
                clean["T"] = clean["T"].where(clean["T"].between(-40, 65))
            if "RH" in clean:
                clean["RH"] = clean["RH"].where(clean["RH"].between(0, 100))
            if "p" in clean:
                clean["p"] = clean["p"].where(clean["p"].between(870, 1085))
            if "O3_raw" in clean:
                clean["O3_raw"] = clean["O3_raw"].where(clean["O3_raw"].between(1, 4094))
            for key in ("CO_raw", "NO2_raw", "PM2_5_raw", "PM10_raw"):
                if key in clean:
                    clean[key] = clean[key].where(clean[key] >= 0)
            group = clean.resample(f"{SAMPLE_SECONDS}s")
            sum_parts.append(group.sum(min_count=1))
            count_parts.append(group.count())
    if not sum_parts:
        raise ValueError(f"No valid timestamps in {src.label()}")
    sums = pd.concat(sum_parts).groupby(level=0).sum(min_count=1)
    counts = pd.concat(count_parts).groupby(level=0).sum()
    out = (sums / counts.where(counts > 0)).sort_index()
    out.index.name = "time_utc"
    print(f"[DATA] {src.basename}: {n_good:,}/{n_total:,} valid rows, "
          f"{len(out):,} time bins")
    return out


def read_reference_gases(src: Source) -> pd.DataFrame:
    with src.open() as f:
        ref = pd.read_csv(f, encoding="utf-8-sig")
    cols = ["timestamp_utc", "CO_ref", "NO2_ref", "O3_ref"]
    if not set(cols).issubset(ref.columns):
        raise ValueError(f"Missing gas reference columns: {set(cols) - set(ref.columns)}")
    ref.index = pd.to_datetime(ref["timestamp_utc"], utc=True, errors="coerce")
    ref = ref[["CO_ref", "NO2_ref", "O3_ref"]].apply(pd.to_numeric, errors="coerce")
    ref = ref[ref.index.notna()]
    ref[ref < 0] = np.nan
    ref = ref.resample(f"{SAMPLE_SECONDS}s").mean()
    ref.index.name = "time_utc"
    return ref


def _excel_date(v) -> date | None:
    if pd.isna(v):
        return None
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(v).date()
    if isinstance(v, (float, int, np.number)):
        return (datetime(1899, 12, 30) + timedelta(days=float(v))).date()
    value = pd.to_datetime(str(v), errors="coerce", dayfirst=True)
    return value.date() if pd.notna(value) else None


def read_reference_pm(src: Source, csv_override: Path | None = None) -> pd.DataFrame:
    # The original file has row 1: "PD Mandria", row 2: PM10/PM2.5,
    # row 3: units, and from row 4: Excel date in column A and daily values in B/C.
    if csv_override is not None:
        d = pd.read_csv(csv_override)
        required = ["day", "PM10_ref", "PM2_5_ref"]
        if not set(required).issubset(d):
            raise ValueError(f"PM CSV must contain columns {required}")
        d["day"] = pd.to_datetime(d["day"]).dt.date
        return d.set_index("day")[["PM2_5_ref", "PM10_ref"]].sort_index()
    try:
        with src.open() as f:
            raw_bytes = f.read()
        if src.basename.lower().endswith(".xls"):
            frame = pd.read_excel(io.BytesIO(raw_bytes), engine="xlrd", header=None)
        else:
            frame = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl", header=None)
    except ImportError as exc:
        raise ImportError("To read the original PM .xls file, install xlrd using: "
                          "python -m pip install xlrd") from exc
    if frame.shape[1] < 3:
        raise ValueError("PM XLS has fewer than three columns")
    rows = []
    for _, row in frame.iloc[3:].iterrows():
        day = _excel_date(row.iloc[0])
        pm10 = pd.to_numeric(row.iloc[1], errors="coerce")
        pm25 = pd.to_numeric(row.iloc[2], errors="coerce")
        if day and 2020 <= day.year <= 2035:
            rows.append({"day": day, "PM10_ref": pm10, "PM2_5_ref": pm25})
    if not rows:
        raise ValueError("No dated daily observations found in PM XLS")
    return pd.DataFrame(rows).drop_duplicates("day").set_index("day").sort_index()


def attach_meteo(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Same colocated station group: ground_screen -> ground_basic
    # Second station group: opc_drone -> gas_drone
    for dst, provider in (("ground_screen", "ground_basic"), ("opc_drone", "gas_drone")):
        for c in ("T", "RH", "p"):
            donor = data[provider][[c]].rename(columns={c: "_donor"})
            if c not in data[dst]:
                data[dst][c] = np.nan
            data[dst][c] = data[dst][c].combine_first(donor["_donor"])
    return data


def get_local_day(idx: pd.DatetimeIndex):
    return idx.tz_convert(TZ).date


def number_of_bins(day: date) -> int:
    # Accounts for daylight-saving time changes.
    a = pd.Timestamp(day).tz_localize(TZ)
    b = pd.Timestamp(day + timedelta(days=1)).tz_localize(TZ)
    return int((b.tz_convert("UTC") - a.tz_convert("UTC")).total_seconds() // SAMPLE_SECONDS)


def usable_gas_days(raw: pd.DataFrame, ref: pd.DataFrame, pollutant: str,
                    instrument: str) -> tuple[pd.DataFrame, list[dict]]:
    columns = GAS_FEATURES_P
    all_values = raw.reindex(columns=columns).join(ref[[f"{pollutant}_ref"]], how="outer")
    all_values["day"] = get_local_day(all_values.index)
    records, kept = [], []
    for day, group in all_values.groupby("day", sort=True):
        if day < DATE_FROM or day > DATE_TO:
            continue
        expected = number_of_bins(day)
        raw_valid = group[columns].notna().all(axis=1)
        ref_valid = group[f"{pollutant}_ref"].notna()
        pair_valid = raw_valid & ref_valid
        raw_cov, ref_cov, pair_cov = [float(mask.sum() / expected)
                                       for mask in (raw_valid, ref_valid, pair_valid)]
        accept = (raw_cov >= MIN_RAW_COVERAGE and ref_cov >= MIN_REF_COVERAGE
                  and pair_cov >= MIN_PAIRED_COVERAGE)
        records.append({"instrument": instrument, "pollutant": pollutant,
                        "day": str(day), "expected_bins": expected,
                        "raw_coverage": raw_cov, "reference_coverage": ref_cov,
                        "paired_coverage": pair_cov, "used": accept})
        if accept:
            good = group.loc[pair_valid, columns + [f"{pollutant}_ref", "day"]].copy()
            good = good.rename(columns={f"{pollutant}_ref": "target"})
            kept.append(good)
    if not kept:
        return pd.DataFrame(columns=columns + ["target", "day"]), records
    return pd.concat(kept).sort_index(), records


def usable_pm_days(raw: pd.DataFrame, ref: pd.DataFrame, pollutant: str,
                   instrument: str) -> tuple[pd.DataFrame, list[dict]]:
    key = f"{pollutant}_raw"
    features = [key, "RH", "T"]
    if any(c not in raw for c in features):
        raise ValueError(f"{instrument}: missing PM or meteorological columns: {features}")
    d = raw[features].copy()
    d["day"] = get_local_day(d.index)
    records, rows = [], []
    for day, group in d.groupby("day", sort=True):
        if day < DATE_FROM or day > DATE_TO:
            continue
        expected = number_of_bins(day)
        valid = group[features].notna().all(axis=1)
        coverage = float(valid.sum() / expected)
        has_ref = day in ref.index and pd.notna(ref.at[day, f"{pollutant}_ref"])
        accept = (coverage >= MIN_RAW_COVERAGE and has_ref)
        records.append({"instrument": instrument, "pollutant": pollutant,
                        "day": str(day), "expected_bins": expected,
                        "raw_coverage": coverage, "reference_coverage": float(has_ref),
                        "paired_coverage": coverage if has_ref else 0., "used": accept})
        if accept:
            means = group.loc[valid, features].mean()
            rows.append({"time_utc": pd.Timestamp(day, tz=TZ).tz_convert("UTC"),
                         "day": day, "target": float(ref.at[day, f"{pollutant}_ref"]),
                         "PM_raw": float(means[key]),
                         "RH": float(means["RH"]), "T": float(means["T"])})
    if not rows:
        return pd.DataFrame(columns=["day", "target", "PM_raw", "RH", "T"]), records
    out = pd.DataFrame(rows).set_index("time_utc").sort_index()
    return out, records


def chronological_day_split(data: pd.DataFrame):
    days = sorted(data.day.unique())
    if len(days) < 6:
        raise ValueError(f"Only {len(days)} complete days found; "
                         "at least 6 days are required for train/validation/test splits")
    train_n = math.floor(0.6 * len(days))
    val_n = max(1, round(0.2 * len(days)))
    train_days = set(days[:train_n])
    val_days = set(days[train_n:train_n + val_n])
    test_days = set(days[train_n + val_n:])
    if not test_days:
        raise ValueError("No days remaining for the test split")
    return {"train": data.loc[data.day.isin(train_days)].copy(),
            "validation": data.loc[data.day.isin(val_days)].copy(),
            "test": data.loc[data.day.isin(test_days)].copy()}


def model_specs(pollutant: str, kind: str):
    if kind == "pm":
        return [("LR", ["PM_raw"]),
                ("MLR_RH", ["PM_raw", "RH"]),
                ("MLR_RH_T", ["PM_raw", "RH", "T"])]
    specs = [("LR", [PRIMARY_GAS[pollutant]]),
             ("MLR_without_pressure", GAS_FEATURES_NO_P),
             ("MLR_with_pressure", GAS_FEATURES_P),
             ("Ridge_without_pressure", GAS_FEATURES_NO_P),
             ("Ridge_with_pressure", GAS_FEATURES_P)]
    for act in ("gelu", "tanh", "swiglu", "relu"):
        specs += [(f"NN_{act}_without_pressure", GAS_FEATURES_NO_P),
                  (f"NN_{act}_with_pressure", GAS_FEATURES_P)]
    return specs


def statistics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    result = {"N": len(y), "R2": np.nan, "Pearson_r": np.nan,
              "Spearman_rho": np.nan, "RMSE": np.nan, "MAE": np.nan,
              "Bias": np.nan, "MedAE": np.nan, "negative_pct": np.nan}
    if not len(y):
        return result
    delta = p - y
    result.update(RMSE=float(np.sqrt(np.mean(delta**2))),
                  MAE=float(np.mean(np.abs(delta))),
                  Bias=float(np.mean(delta)),
                  MedAE=float(np.median(np.abs(delta))),
                  negative_pct=float(100 * np.mean(p < 0)))
    if len(y) >= 2 and np.std(y) > 0:
        result["R2"] = float(r2_score(y, p))
        if np.std(p) > 0:
            result["Pearson_r"] = float(stats.pearsonr(y, p).statistic)
            result["Spearman_rho"] = float(stats.spearmanr(y, p).statistic)
    return result


def build_linear(which: str, Xtrain, ytrain, Xval, yval):
    alphas = (0.01, 0.1, 1., 10., 100.) if which.startswith("Ridge") else (None,)
    best, best_rmse, best_note = None, float("inf"), ""
    for alpha in alphas:
        pipe = Pipeline([("standardize", StandardScaler()),
                         ("model", Ridge(alpha=alpha) if alpha is not None
                          else LinearRegression())])
        pipe.fit(Xtrain, ytrain)
        rmse = float(np.sqrt(mean_squared_error(yval, pipe.predict(Xval))))
        if rmse < best_rmse:
            best, best_rmse = pipe, rmse
            best_note = f"alpha={alpha}" if alpha is not None else ""
    return best, best_note


def build_neural(Xtrain, ytrain, Xval, yval, activation, epochs, seed):
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("Neural networks require PyTorch. Install it in your active Spyder Python environment with: python -m pip install torch") from exc
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    sx = StandardScaler().fit(Xtrain)
    xtrain = sx.transform(Xtrain).astype("float32")
    xval = sx.transform(Xval).astype("float32")
    ymean, ystd = float(ytrain.mean()), float(ytrain.std())
    if ystd < 1e-10:
        raise ValueError("Training targets have zero variance")
    yytrain = ((ytrain - ymean) / ystd).astype("float32")
    yyval = ((yval - ymean) / ystd).astype("float32")
    dim, hidden = Xtrain.shape[1], 8

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hid = nn.Linear(dim, hidden * 2 if activation == "swiglu" else hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            x = self.hid(x)
            if activation == "swiglu":
                a, b = x.chunk(2, dim=-1)
                x = F.silu(a) * b
            elif activation == "gelu":
                x = F.gelu(x)
            elif activation == "relu":
                x = F.relu(x)
            else:
                x = torch.tanh(x)
            return self.out(x).flatten()

    net = Net()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    a, b = torch.from_numpy(xtrain), torch.from_numpy(yytrain)
    av, bv = torch.from_numpy(xval), torch.from_numpy(yyval)
    generator = torch.Generator().manual_seed(seed)
    best_params, best_val, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        net.train()
        for idx in torch.randperm(len(a), generator=generator).split(512):
            optim.zero_grad()
            loss = loss_fn(net(a[idx]), b[idx])
            loss.backward()
            optim.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(av), bv))
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_params = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 16:
                break
    net.load_state_dict(best_params)
    net.eval()

    class Predictor:
        def predict(self, X):
            x = torch.from_numpy(sx.transform(np.asarray(X)).astype("float32"))
            with torch.no_grad():
                p = net(x).detach().cpu().numpy()
            return p.astype(float) * ystd + ymean

    return Predictor(), f"hidden=8;best_epoch={best_epoch};activation={activation};seed={seed}"


def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_all_model_test_timeseries(parts, estimators, val_errors, instrument,
                                    pollutant, dest: Path, kind: str) -> None:
    """Plot the held-out reference against EVERY fitted calibration method.

    Gas instruments are compared with the Thermo gas reference at identical
    30-second timestamps; PM instruments use the original daily PM reference.
    The existing train/validation/test split is preserved (no refitting here).
    """
    import matplotlib.dates as mdates

    test = parts["test"].sort_index()
    if test.empty:
        return
    dest.mkdir(parents=True, exist_ok=True)
    unit = {"CO": "ppm", "NO2": "ppb", "O3": "ppb",
            "PM2_5": "µg/m³", "PM10": "µg/m³"}.get(pollutant, "reference units")
    gas = kind == "gas"
    reference_name = "Thermo reference" if gas else "PM reference (daily mean)"
    best_name = min(val_errors, key=val_errors.get)  # validation RMSE, never test selection
    y_ref = test.target.to_numpy(float)

    # Predictions are made only for the unchanged, held-out test observations.
    predictions = {}
    for name, (model, features) in estimators.items():
        predictions[name] = np.asarray(
            model.predict(test[features].to_numpy(float)), dtype=float).reshape(-1)

    # Respect local day boundaries and do not connect separate test days.
    # The source timestamps remain in UTC for gases; PM uses the local day.
    x_times = test.index if gas else test.index.tz_convert(TZ)
    groups = [np.flatnonzero(test.day.to_numpy() == day)
              for day in sorted(test.day.unique())]
    cmap = plt.get_cmap("tab20")
    other_methods = [name for name in estimators if name != best_name]
    method_colors = {name: cmap(i / max(len(other_methods), 1))
                     for i, name in enumerate(other_methods)}
    method_colors[best_name] = "#d65b34"
    ref_color = "#151f30"
    small_markers = {"marker": "o", "markersize": 5} if not gas else {}
    ref_label = f"{reference_name}"
    summary = "30-second averages" if gas else "daily averages"
    pollutant_display = pollutant.replace("PM2_5", "PM₂.₅").replace("PM10", "PM₁₀").replace("NO2", "NO₂").replace("O3", "O₃")

    # A) All models OVERLAID, in one panel, with a strong reference line.
    fig, ax = plt.subplots(figsize=(17, 7))
    for name in other_methods:
        for j, idx in enumerate(groups):
            ax.plot(x_times[idx], predictions[name][idx],
                    color=method_colors[name], linewidth=1.05, alpha=.72,
                    label=name.replace("_", " ") if j == 0 else "_nolegend_",
                    **small_markers)
    for j, idx in enumerate(groups):
        ax.plot(x_times[idx], predictions[best_name][idx],
                color=method_colors[best_name], linewidth=2.25, alpha=.97,
                label=f"Best validation: {best_name.replace('_', ' ')}" if j == 0 else "_nolegend_",
                zorder=5, **small_markers)
        ax.plot(x_times[idx], y_ref[idx], color=ref_color, linewidth=2.4,
                label=ref_label if j == 0 else "_nolegend_", zorder=7,
                **small_markers)

    ax.set_title(f"{instrument} | {pollutant_display}: all calibration models vs reference | TEST",
                 fontsize=15, pad=15)
    ax.set_ylabel(f"{pollutant_display} concentration ({unit})", fontsize=12)
    ax.set_xlabel("UTC" if gas else f"Local date ({TZ})", fontsize=11)
    ax.grid(alpha=.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    # Keep method names visible without hiding the time series.
    handles, labels = ax.get_legend_handles_labels()
    # Reference and validation-selected model are placed first.
    ordering = [labels.index(ref_label), labels.index(f"Best validation: {best_name.replace('_', ' ')}")]
    ordering += [i for i in range(len(labels)) if i not in ordering]
    ax.legend([handles[i] for i in ordering], [labels[i] for i in ordering],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              frameon=False, title="Test predictions")
    fig.text(.055, .017,
             f"Held-out test days | {summary} | model selection based on validation RMSE",
             fontsize=9, color="#566170")
    fig.subplots_adjust(left=.065, right=.73, top=.89, bottom=.15)
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"test_timeseries_all_models.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    # B) Readable alternative: every method gets its OWN panel next to the
    # same reference. It is easier to compare many overlapping models.
    ncols = 3 if len(predictions) >= 3 else len(predictions)
    nrows = math.ceil(len(predictions) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, nrows * 2.8),
                             squeeze=False, sharex=True, sharey=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (name, pred) in zip(axes.flat, predictions.items()):
        ax.set_visible(True)
        color = method_colors[name]
        for idx in groups:
            ax.plot(x_times[idx], y_ref[idx], color=ref_color,
                    linewidth=1.65, **small_markers)
            ax.plot(x_times[idx], pred[idx], color=color,
                    linewidth=1.5, **small_markers)
        score = statistics(y_ref, pred)["RMSE"]
        heading = name.replace("_", " ") + ("  ★" if name == best_name else "")
        ax.set_title(f"{heading}\nTest RMSE: {score:.2f} {unit}", fontsize=10.5,
                     fontweight="bold" if name == best_name else "normal")
        ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{instrument} | {pollutant_display} | all models vs {reference_name} | held-out test",
                 fontsize=16, y=.995)
    fig.text(.5, .006, f"Black: {reference_name}    |    Colored: model prediction    |    ★ selected by validation RMSE",
             ha="center", fontsize=10)
    fig.supxlabel("UTC" if gas else f"Local date ({TZ})", fontsize=11, y=.025)
    fig.supylabel(f"{pollutant_display} ({unit})", fontsize=11, x=.006)
    fig.tight_layout(rect=(.02, .045, 1, .975), h_pad=1.6, w_pad=1.2)
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"test_timeseries_each_model.{suffix}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOTS] {instrument}/{pollutant}: {len(predictions)} models + {reference_name} -> {dest}")


def diagnostic_figures(parts, estimators, val_errors, instrument, pollutant,
                       feature_names, dest: Path, plots: str, kind: str = "gas"):
    if plots == "none":
        return
    dest.mkdir(parents=True, exist_ok=True)
    # New: reference + ALL models overlaid on one test time series;
    # also export individually faceted comparisons (PNG and vector PDF).
    plot_all_model_test_timeseries(parts, estimators, val_errors,
                                   instrument, pollutant, dest, kind)
    bestname = min(val_errors, key=val_errors.get)  # validation data only
    model, features = estimators[bestname]
    test, train = parts["test"], parts["train"]
    ytest = test.target.to_numpy(dtype=float)
    ptest = model.predict(test[features].to_numpy(dtype=float))

    # 1. Time series over the held-out test period
    plt.figure(figsize=(12, 4))
    plt.plot(test.index, ytest, label="Reference", lw=1.2)
    plt.plot(test.index, ptest, label=bestname, lw=1.2)
    plt.xlabel("UTC")
    plt.ylabel("Reference unit (as in source file)")
    plt.title(f"{instrument} | {pollutant} | Test | validation-best={bestname}")
    plt.legend()
    save_plot(dest / "test_timeseries.png")

    # 2. All methods evaluated on the identical held-out test set
    fig, axes = plt.subplots(nrows=math.ceil(len(estimators) / 3), ncols=3,
                             figsize=(12, 4 * math.ceil(len(estimators) / 3)), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (name, (mod, feat)) in zip(axes.flat, estimators.items()):
        ax.set_visible(True)
        prediction = mod.predict(test[feat].to_numpy(dtype=float))
        ax.scatter(ytest, prediction, s=4 if len(ytest) > 100 else 18, alpha=0.3)
        lo, hi = float(min(ytest.min(), prediction.min())), float(max(ytest.max(), prediction.max()))
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", lw=1)
        ax.set(title=f"{name} | RMSE={statistics(ytest, prediction)['RMSE']:.3g}",
               xlabel="Reference", ylabel="Prediction")
    save_plot(dest / "test_scatter_all_models.png")

    # 3. Performance metrics
    fig, ax = plt.subplots(figsize=(max(9, len(estimators) * 0.8), 4.3))
    names = list(estimators)
    xs = np.arange(len(names))
    ax.bar(xs - 0.18, [val_errors[m] for m in names], width=0.36, label="Validation-RMSE")
    ax.bar(xs + 0.18,
           [statistics(ytest, estimators[m][0].predict(test[estimators[m][1]].to_numpy(float)))["RMSE"]
            for m in names], width=0.36, label="Test-RMSE")
    ax.set_xticks(xs, names, rotation=65, ha="right")
    ax.set_ylabel("RMSE (reference unit)")
    ax.set_title(f"{instrument} | {pollutant} | Model comparison")
    ax.legend()
    save_plot(dest / "validation_vs_test_RMSE.png")

    if plots == "basic":
        return
    # 4. Dependency plots: observed data versus input; PDP of validation-selected model,
    #    other features held at their training-set median values.
    subset = train.iloc[np.linspace(0, len(train) - 1, min(4000, len(train))).astype(int)]
    med = train[features].median()
    for feature in feature_names:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(subset[feature], subset.target, alpha=0.16, s=8, label="Training: measured")
        if feature in features:
            qlo, qhi = train[feature].quantile([0.05, 0.95])
            if np.isfinite(qlo) and np.isfinite(qhi) and qlo < qhi:
                grid = np.linspace(qlo, qhi, 120)
                X = pd.DataFrame({c: np.repeat(med[c], len(grid)) for c in features})
                X[feature] = grid
                ax.plot(grid, model.predict(X[features].to_numpy(float)), lw=2.2,
                        label=f"PDP: {bestname}")
        ax.set(title=f"{instrument} | {pollutant}: Dependence on {feature}",
               xlabel=feature, ylabel="Reference / prediction")
        ax.legend()
        save_plot(dest / f"dependency_{feature}.png")

    # 5. Correlation matrix for the training set (descriptive only)
    cm = train[feature_names + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(cm) * 0.9), max(5, len(cm) * 0.7)))
    im = ax.imshow(cm.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cm)), list(cm), rotation=60, ha="right")
    ax.set_yticks(range(len(cm)), list(cm))
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(f"{instrument} | {pollutant}: Training correlations")
    save_plot(dest / "train_correlation_matrix.png")


def compare_one(data: pd.DataFrame, instrument: str, pollutant: str, kind: str,
                output: Path, args):
    parts = chronological_day_split(data)
    specs = model_specs(pollutant, kind)
    if args.only:
        requested = set(x.strip() for x in args.only.split(","))
        specs = [sp for sp in specs if sp[0] in requested or sp[0].split("_")[0] in requested]
    if not specs:
        raise ValueError(f"No methods selected for {pollutant}")
    estimators, val_errors, metrics_rows, prediction_rows = {}, {}, [], []
    source_features = GAS_FEATURES_P if kind == "gas" else ["PM_raw", "RH", "T"]
    calibrated = data[["day", "target"] + source_features].copy()
    calibrated = calibrated.rename(columns={"target": "reference"})
    calibrated.insert(1, "phase", "")
    for phase, frame in parts.items():
        calibrated.loc[frame.index, "phase"] = phase
    for method, features in specs:
        xtrain = parts["train"][features].to_numpy(float)
        ytrain = parts["train"].target.to_numpy(float)
        xval = parts["validation"][features].to_numpy(float)
        yval = parts["validation"].target.to_numpy(float)
        if method.startswith("NN_"):
            activation = method.split("_")[1]
            model, info = build_neural(xtrain, ytrain, xval, yval, activation,
                                       args.epochs, args.seed)
        else:
            model, info = build_linear(method, xtrain, ytrain, xval, yval)
        estimators[method] = (model, features)
        val_errors[method] = statistics(yval, model.predict(xval))["RMSE"]
        for phase, frame in parts.items():
            pred = model.predict(frame[features].to_numpy(float))
            calibrated.loc[frame.index, f"pred_{method}"] = pred
            m = statistics(frame.target.to_numpy(float), pred)
            metrics_rows.append(dict(instrument=instrument, pollutant=pollutant,
                                     dataset=kind, model=method, phase=phase,
                                     features="|".join(features), parameters=info,
                                     days="|".join(map(str, sorted(frame.day.unique()))),
                                     n_days=len(frame.day.unique()), **m))
            if phase == "test":
                prediction_rows.append(pd.DataFrame({
                    "instrument": instrument, "pollutant": pollutant, "model": method,
                    "time_utc": frame.index.astype(str), "day": frame.day.astype(str).to_numpy(),
                    "reference": frame.target.to_numpy(float), "prediction": pred,
                }))
    best = min(val_errors, key=val_errors.get)
    wide = calibrated.reset_index().rename(columns={"index": "time_utc"})
    write_csv(wide, output / "calibrated_series" / kind /
              f"{instrument}_{pollutant}.csv")
    dest = output / "plots" / kind / instrument / pollutant
    feat_names = GAS_FEATURES_P if kind == "gas" else ["PM_raw", "RH", "T"]
    diagnostic_figures(parts, estimators, val_errors, instrument, pollutant,
                       feat_names, dest, args.plots, kind)
    print(f"[MODELS] {instrument}/{pollutant}: "
          f"Days train/validation/test: {len(parts['train'].day.unique())}/"
          f"{len(parts['validation'].day.unique())}/"
          f"{len(parts['test'].day.unique())}; "
          f"{len(specs)} models; best on validation: {best}")
    return metrics_rows, prediction_rows, {
        "instrument": instrument, "pollutant": pollutant, "kind": kind,
        "best_by_validation_RMSE": best,
        "best_validation_RMSE": val_errors[best],
        "train_days": [str(x) for x in sorted(parts["train"].day.unique())],
        "validation_days": [str(x) for x in sorted(parts["validation"].day.unique())],
        "test_days": [str(x) for x in sorted(parts["test"].day.unique())],
    }


def write_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")


def main():
    parser = argparse.ArgumentParser(description="Mandria: original data calibration and model comparison")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--zip", type=Path, default=None, help="Use ZIP instead of extracted directories")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: data-root/results_calibration)")
    parser.add_argument("--check", action="store_true", help="List source files and columns; do not train models")
    parser.add_argument("--skip-pm", action="store_true", help="Gas calibration only")
    parser.add_argument("--skip-gas", action="store_true", help="PM calibration only")
    parser.add_argument("--pm-reference-csv", type=Path, default=None,
                        help="Optional daily-reference CSV extracted from the original XLS")
    parser.add_argument("--instruments", default="ground_basic,ground_screen,gas_drone,opc_drone",
                        help="Comma-separated list")
    parser.add_argument("--gases", default="CO,NO2,O3", help="Comma-separated: CO,NO2,O3")
    parser.add_argument("--only", default=None, help="Quick test, e.g. LR,MLR,Ridge")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", choices=["none", "basic", "full"], default="full")
    args = parser.parse_args()
    sources = find_inputs(args.data_root, args.zip)
    for key, src in sources.items():
        print(f"[FILE] {key:20s}: {src.label()}")
    if args.check:
        for key, src in sources.items():
            if key != "pm_reference":
                print(f"[COLUMNS] {key}: {', '.join(csv_header(src))}")
        if not args.skip_pm:
            try:
                pm = read_reference_pm(sources["pm_reference"], args.pm_reference_csv)
                print(f"[PM DAYS] {len(pm)}: {pm.index.min()} to {pm.index.max()}")
            except ImportError as exc:
                print(f"[NOTE] {exc}")
        print("Input check complete. No models were trained.")
        return
    out = args.output or (args.data_root / "results_calibration")
    out.mkdir(parents=True, exist_ok=True)
    selected = {x.strip() for x in args.instruments.split(",")}
    pollutants = [p.strip().upper() for p in args.gases.split(",")]
    if not set(pollutants).issubset(PRIMARY_GAS):
        raise ValueError("--gases only accepts CO,NO2,O3")
    gas_instruments = sorted(selected & {"ground_basic", "ground_screen", "gas_drone"})
    pm_instruments = sorted(selected & {"ground_basic", "ground_screen", "opc_drone"})
    to_read = set()
    if not args.skip_gas:
        to_read.update(gas_instruments)
    if not args.skip_pm:
        to_read.update(pm_instruments)
    if "ground_screen" in to_read:
        to_read.add("ground_basic")
    if "opc_drone" in to_read:
        to_read.add("gas_drone")
    all_sensor_data = {name: read_sensor_binned(sources[name]) for name in sorted(to_read)}
    if "ground_screen" in all_sensor_data or "opc_drone" in all_sensor_data:
        attach_meteo(all_sensor_data)
    gas_ref = read_reference_gases(sources["thermo_reference"]) if not args.skip_gas else None
    pm_ref = read_reference_pm(sources["pm_reference"], args.pm_reference_csv) if not args.skip_pm else None
    coverage, metric_tables, test_tables, selections = [], [], [], []

    if gas_ref is not None:
        for inst in gas_instruments:
            for pollutant in pollutants:
                data, cov = usable_gas_days(all_sensor_data[inst], gas_ref, pollutant, inst)
                coverage.extend(cov)
                if data.empty:
                    warnings.warn(f"{inst}/{pollutant}: no days passed the completeness checks")
                    continue
                try:
                    rows, preds, meta = compare_one(data, inst, pollutant, "gas", out, args)
                    metric_tables.extend(rows)
                    test_tables.extend(preds)
                    selections.append(meta)
                except ValueError as exc:
                    warnings.warn(f"{inst}/{pollutant}: {exc}")
    if pm_ref is not None:
        for inst in pm_instruments:
            for pollutant in ("PM2_5", "PM10"):
                data, cov = usable_pm_days(all_sensor_data[inst], pm_ref, pollutant, inst)
                coverage.extend(cov)
                if data.empty:
                    warnings.warn(f"{inst}/{pollutant}: no days passed the completeness checks")
                    continue
                try:
                    rows, preds, meta = compare_one(data, inst, pollutant, "pm", out, args)
                    metric_tables.extend(rows)
                    test_tables.extend(preds)
                    selections.append(meta)
                except ValueError as exc:
                    warnings.warn(f"{inst}/{pollutant}: {exc}")
    write_csv(pd.DataFrame(coverage), out / "data_coverage_by_day.csv")
    if metric_tables:
        scores = pd.DataFrame(metric_tables)
        write_csv(scores, out / "all_model_metrics.csv")
        write_csv(scores.query("phase == 'test'").sort_values(
            ["dataset", "instrument", "pollutant", "RMSE"]), out / "test_model_ranking.csv")
    if test_tables:
        write_csv(pd.concat(test_tables, ignore_index=True), out / "all_test_predictions.csv")
    if selections:
        write_csv(pd.DataFrame(selections), out / "validation_best_models.csv")
    metadata = {
        "provenance": {name: src.label() for name, src in sources.items()},
        "timezone": TZ, "seconds_per_bin": SAMPLE_SECONDS,
        "coverage": {"min_raw": MIN_RAW_COVERAGE,
                     "min_ref": MIN_REF_COVERAGE, "min_paired": MIN_PAIRED_COVERAGE},
        "seed": args.seed, "epochs": args.epochs,
        "meteorology_transfer": {"ground_screen": "ground_basic",
                                 "opc_drone": "gas_drone"},
        "sensor_mapping": RAW_MAP,
        "split": "chronological calendar days 60/20/20; no days cross split boundaries",
        "selection": "validation RMSE only; test is never used for selection",
        "note": "PM: approx. 10 daily reference values; use metrics with great caution",
    }
    (out / "methodology_and_sources.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCOMPLETE: {out.resolve()}")
    print("Output files: data_coverage_by_day.csv, all_model_metrics.csv, "
          "test_model_ranking.csv, all_test_predictions.csv, "
          "calibrated_series/, plots/")


if __name__ == "__main__":
    main()
