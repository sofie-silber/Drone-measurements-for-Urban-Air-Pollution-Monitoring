#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Mandria calibrations to three field campaigns at genuine 1-second resolution.

CAMPAIGNS
  2026-09-22: Rubano church + Bosco roundabout (Padua region, Italy)
  2026-09-23: Rubano roundabout (Padua region, Italy)
  2026-10-21: Munich - visiting Graz researchers (LOG008 + LOG009)

Transfer uses the same model specification and training functions as
calibration_paris_english.py, on original Mandria reference data. No use of flight
samples for fitting. Flight predictions are strictly exploratory: gas models were
trained on 30-second means and PM models on daily averages.

Never clip or replace a negative prediction before negative-value screening.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import calibration_paris_english as cal
except ImportError as exc:
    raise ImportError("Place calibration_paris_english.py next to this script.") from exc

DEFAULT_CALIBRATION_ROOT = Path(r"C:\Users\xy-84\Desktop\Master\Thesis\Background_Data_Paris")
DEFAULT_FLIGHTS_ROOT = Path(r"C:\Users\xy-84\Desktop\Master\Thesis")
TZ = "Europe/Rome"                 # Same UTC offset/DST rules as Munich on the flight dates
SAMPLE_RULE = "1s"
MIN_SHORTLIST_COVERAGE = 0.95       # Require predictions for >=95% of valid raw samples PER SESSION
GAS_COLS = ["CO_raw", "NO2_raw", "O3_raw", "T", "RH", "p"]
PM_COLS = ["PM2_5_raw", "PM10_raw", "T", "RH"]
SENSOR_MAP = {
    "mics_reducing_ohm": "CO_raw", "mics_oxidising_ohm": "NO2_raw",
    "mq131_raw": "O3_raw", "pm2_5_ug_m3": "PM2_5_raw",
    "pm10_ug_m3": "PM10_raw", "temp_c": "T",
    "humidity_percent": "RH", "pressure_hpa": "p",
    "gps_lat": "latitude", "gps_lon": "longitude", "gps_alt_m": "gps_alt_m",
    "gps_speed_kmh": "speed_kmh", "gps_fix": "gps_fix",
    "gps_hdop": "gps_hdop", "gps_satellites": "gps_satellites",
}
PRIMARY_RAW = {"CO": "CO_raw", "NO2": "NO2_raw", "O3": "O3_raw",
               "PM2_5": "PM2_5_raw", "PM10": "PM10_raw"}
CAMPAIGNS = {
    "padua_2026_09_22": {"name": "Padua: Rubano church and Bosco roundabout", "date": "2026-09-22"},
    "padua_2026_09_23": {"name": "Padua: Rubano roundabout", "date": "2026-09-23"},
    "munich_2026_10_21": {"name": "Munich: Visiting Graz researchers", "date": "2026-10-21"},
}


@dataclass(frozen=True)
class InputFile:
    name: str
    archive: Path | None = None
    actual_path: Path | None = None

    def read(self) -> pd.DataFrame:
        if self.archive is not None:
            with zipfile.ZipFile(self.archive) as archive:
                with archive.open(self.name) as source:
                    return pd.read_csv(source, low_memory=False, encoding="utf-8-sig")
        if self.actual_path is None:
            raise RuntimeError("Missing file location")
        return pd.read_csv(self.actual_path, low_memory=False, encoding="utf-8-sig")

    @property
    def provenance(self) -> str:
        return f"{self.archive}::{self.name}" if self.archive else str(self.actual_path)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def find_file(filename: str, root: Path) -> Path:
    """Prefer files next to the script, then recurse under the configured flight root."""
    here = Path(__file__).resolve().parent
    for f in (here / filename, root / filename):
        if f.is_file():
            return f.resolve()
    if root.is_dir():
        matches = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() == filename.lower())
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(
        f"Could not find {filename}. Place it next to this script or set --flights-root."
    )


def flight_sources(args) -> dict[str, list[InputFile]]:
    names = {
        "padua_2026_09_22": "zip_raw_data_22.9.bosco_rubano.zip",
        "padua_2026_09_23": "raw_data_23.9.26_rubano_roundabout.zip",
    }
    archive_paths = {
        "padua_2026_09_22": args.zip_22 or find_file(names["padua_2026_09_22"], args.flights_root),
        "padua_2026_09_23": args.zip_23 or find_file(names["padua_2026_09_23"], args.flights_root),
    }
    sources: dict[str, list[InputFile]] = {}
    for key, archive_path in archive_paths.items():
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as z:
            contents = [InputFile(member, archive=archive_path)
                        for member in z.namelist()
                        if member.lower().endswith(".csv") and
                        Path(member).name.lower().startswith(("gas_drone", "opc_drone", "ground_basic"))]
        if not contents:
            raise ValueError(f"No drone or ground weather CSVs in {archive_path}")
        sources[key] = contents
    sources["munich_2026_10_21"] = [
        InputFile(name, actual_path=(args.log_008 if name == "LOG008.CSV" else args.log_009)
                  or find_file(name, args.flights_root)) for name in ("LOG008.CSV", "LOG009.CSV")
    ]
    return sources


def utc_window(day: str, start_local: str, end_local: str):
    return (pd.Timestamp(f"{day} {start_local}", tz=TZ).tz_convert("UTC"),
            pd.Timestamp(f"{day} {end_local}", tz=TZ).tz_convert("UTC"))


def mission_definitions() -> list[dict[str, Any]]:
    """Local-time flight intervals manually transcribed from uploaded campaign notes.

    The Munich files lack annotated takeoff/landing times; they are mission logs,
    NOT independently confirmed continuous in-air periods.
    """
    events = [
        ("padua_2026_09_22", "Rubano church: OPC flight", "opc_drone", "11:37:20", "11:42:35"),
        ("padua_2026_09_22", "Rubano church: gas flight", "gas_drone", "12:09:44", "12:17:40"),
        ("padua_2026_09_22", "Bosco roundabout: OPC flight", "opc_drone", "13:13:20", "13:20:43"),
        ("padua_2026_09_22", "Bosco roundabout: gas flight", "gas_drone", "13:25:05", "13:32:05"),
        ("padua_2026_09_23", "Rubano 5 m: gas flight", "gas_drone", "17:55:10", "18:02:05"),
        ("padua_2026_09_23", "Rubano 5 m: OPC flight", "opc_drone", "18:10:30", "18:17:25"),
        ("padua_2026_09_23", "Rubano 10 m: OPC flight", "opc_drone", "18:57:25", "19:01:15"),
        ("padua_2026_09_23", "Rubano 10 m: gas flight", "gas_drone", "19:08:30", "19:12:05"),
    ]
    rows = []
    for campaign, label, sensor, start, end in events:
        a, b = utc_window(CAMPAIGNS[campaign]["date"], start, end)
        rows.append({"campaign": campaign, "session": slug(label), "label": label,
                     "sensor": sensor, "start_utc": a, "end_utc": b,
                     "window_source": "supplied field notes (local CEST)"})
    for fname in ("LOG008.CSV", "LOG009.CSV"):
        rows.append({"campaign": "munich_2026_10_21", "session": slug(fname),
                     "label": f"Munich mission log {fname}", "sensor": "gas_drone",
                     "start_utc": None, "end_utc": None,
                     "source_match": fname, "window_source": "entire valid log; takeoff/landing not annotated"})
    return rows


def parse_flight_file(source: InputFile, day: str) -> pd.DataFrame:
    frame = source.read()
    needed = {"gps_date", "gps_time_utc"}
    if not needed.issubset(frame):
        raise ValueError(f"Missing GPS timestamp columns in {source.provenance}")
    times = pd.to_datetime(frame["gps_date"].astype("string").str.strip() + " " +
                           frame["gps_time_utc"].astype("string").str.strip(),
                           errors="coerce", utc=True, format="mixed")
    valid = times.dt.strftime("%Y-%m-%d").eq(day).fillna(False).to_numpy(bool)
    if not valid.any():
        raise ValueError(f"No valid {day} UTC GPS dates in {source.provenance}")
    trimmed = frame.loc[valid].copy()
    out = pd.DataFrame(index=pd.DatetimeIndex(times[valid], name="time_utc"))
    for original, new in SENSOR_MAP.items():
        if original in trimmed:
            out[new] = pd.to_numeric(trimmed[original].to_numpy(), errors="coerce")
    out["source_file"] = Path(source.name).name
    if "T" in out:
        out["T"] = out["T"].where(out["T"].between(-40, 65))
    if "RH" in out:
        out["RH"] = out["RH"].where(out["RH"].between(0, 100))
    if "p" in out:
        out["p"] = out["p"].where(out["p"].between(870, 1085))
    if "O3_raw" in out:
        out["O3_raw"] = out["O3_raw"].where(out["O3_raw"].between(1, 4094))
    for c in ("CO_raw", "NO2_raw", "PM2_5_raw", "PM10_raw"):
        if c in out:
            out[c] = out[c].where(out[c] >= 0)
    out = out.sort_index()
    print(f"[FLIGHT DATA] {source.name}: {len(out):,}/{len(frame):,} rows with valid {day} timestamps")
    return out


def one_second(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean of observed rows in the same UTC second; never interpolate gaps."""
    if frame.empty:
        return frame
    numerics = frame.select_dtypes(include="number").columns
    resampled = frame[numerics].resample(SAMPLE_RULE).mean()
    if "gps_fix" in resampled:
        # A reliable GPS coordinate requires an actual fix, not a bin-mean fix.
        fixed = frame["gps_fix"].resample(SAMPLE_RULE).max()
        resampled["gps_fix"] = fixed
        for col in ("latitude", "longitude", "gps_alt_m", "gps_hdop"):
            if col in resampled:
                resampled[col] = resampled[col].where(resampled["gps_fix"] == 1)
    resampled["source_file"] = frame["source_file"].resample(SAMPLE_RULE).first()
    return resampled


def build_campaign_frames(sources: dict, intervals: list[dict]):
    output: dict[str, list[tuple[dict, pd.DataFrame]]] = {}
    for key, inputfiles in sources.items():
        day = CAMPAIGNS[key]["date"]
        content: dict[str, list[pd.DataFrame]] = {"gas_drone": [], "opc_drone": [], "ground_basic": []}
        by_file: dict[str, pd.DataFrame] = {}
        for f in inputfiles:
            name = Path(f.name).name.lower()
            sensor = ("opc_drone" if name.startswith("opc_drone") else
                      "ground_basic" if name.startswith("ground_basic") else "gas_drone")
            try:
                frame = parse_flight_file(f, day)
            except ValueError as err:
                warnings.warn(str(err))
                continue
            content[sensor].append(frame)
            by_file[Path(f.name).name.upper()] = frame
        stacked = {}
        for sensor, pieces in content.items():
            if pieces:
                all_rows = pd.concat(pieces).sort_index()
                all_rows = all_rows.loc[~all_rows.index.duplicated(keep="last")]
                stacked[sensor] = one_second(all_rows)
        sessions = []
        for event in [v for v in intervals if v["campaign"] == key]:
            if event.get("source_match"):
                fname = event["source_match"].upper()
                if fname not in by_file:
                    raise ValueError(f"Missing Munich log {fname}")
                part = one_second(by_file[fname])
            else:
                if event["sensor"] not in stacked:
                    raise ValueError(f"Missing {event['sensor']} for {event['session']}")
                part = stacked[event["sensor"]].loc[event["start_utc"]:event["end_utc"]].copy()
            if part.empty:
                raise ValueError(f"No measured timestamps for flight interval {event['label']}")
            if event["sensor"] == "opc_drone":
                # Ground meteorology is time-matched, not co-measured by the flying OPC.
                # A gas drone can supply contemporaneous data only when it actually overlaps.
                donors = []
                for donor_name in ("ground_basic", "gas_drone"):
                    donor = stacked.get(donor_name)
                    if donor is not None and {"T", "RH", "p"}.issubset(donor):
                        donors.append((donor_name, donor[["T", "RH", "p"]]))
                part = merge_meteo(part, donors, tolerance_seconds=5)
            else:
                part["meteo_source"] = "onboard_gas_drone"
            part["campaign"] = key
            part["session"] = event["session"]
            part["sensor"] = event["sensor"]
            part["time_local"] = part.index.tz_convert(TZ).astype(str)
            part["gps_alt_relative_m"] = np.nan
            if "gps_alt_m" in part:
                valid_alt = part["gps_alt_m"].dropna()
                if len(valid_alt) > 5:
                    # Relative GPS altitude is NOT validated height above ground.
                    part["gps_alt_relative_m"] = part["gps_alt_m"] - valid_alt.quantile(0.05)
            print(f"[SESSION] {key}/{event['label']}: {len(part):,} 1-second bins")
            sessions.append((event, part))
        output[key] = sessions
    return output


def merge_meteo(frame: pd.DataFrame, donors, tolerance_seconds: int) -> pd.DataFrame:
    out = frame.copy()
    for name in ("T", "RH", "p"):
        if name not in out:
            out[name] = np.nan
    out["meteo_source"] = "missing"
    for provider, meteo in donors:
        source = meteo.dropna(how="all")
        if source.empty:
            continue
        # merge_asof on time ensures no use of unrelated observations separated by minutes.
        left = pd.DataFrame({"time_utc": out.index}).sort_values("time_utc")
        right = source.reset_index().sort_values("time_utc")
        merged = pd.merge_asof(left, right, on="time_utc", direction="nearest",
                               tolerance=pd.Timedelta(seconds=tolerance_seconds))
        take = out["T"].isna() & merged["T"].notna().to_numpy()
        for col in ("T", "RH", "p"):
            dest_missing = out[col].isna()
            out.loc[dest_missing, col] = merged.loc[dest_missing.to_numpy(), col].to_numpy()
        out.loc[take, "meteo_source"] = provider
    return out


def prepare_training(args):
    """Refit the exact same algorithms as the supplied calibration script.

    Splits are whole calibration days; flights never contribute to training.
    """
    files = cal.find_inputs(args.calibration_root, args.calibration_zip)
    if args.check:
        for label, ref in sorted(files.items()):
            print(f"[CALIBRATION INPUT] {label}: {ref.label()}")
        return {}
    gas = cal.read_sensor_binned(files["gas_drone"])
    pm = cal.read_sensor_binned(files["opc_drone"])
    grouped = {"gas_drone": gas, "opc_drone": pm}
    # The original calibration borrows meteorology from colocated gas_drone.
    # Its helper expects all four sensor datasets, so implement just the
    # instrument-matched OPC-to-gas step here, preserving the same time bins.
    for col in ("T", "RH", "p"):
        if col not in pm:
            pm[col] = np.nan
        pm[col] = pm[col].combine_first(gas[col].rename(col))
    gas_ref = cal.read_reference_gases(files["thermo_reference"])
    pm_ref = cal.read_reference_pm(files["pm_reference"], args.pm_reference_csv)
    fitted = {}
    rows = []
    for sensor, kind, targets in (
        ("gas_drone", "gas", ("CO", "NO2", "O3")),
        ("opc_drone", "pm", ("PM2_5", "PM10")),
    ):
        for pollutant in targets:
            if args.pollutants and pollutant not in args.pollutants:
                continue
            if kind == "gas":
                dataset, coverage = cal.usable_gas_days(grouped[sensor], gas_ref, pollutant, sensor)
            else:
                dataset, coverage = cal.usable_pm_days(grouped[sensor], pm_ref, pollutant, sensor)
            if dataset.empty:
                raise ValueError(f"No valid calibration days for {sensor}/{pollutant}")
            split = cal.chronological_day_split(dataset)
            candidates = cal.model_specs(pollutant, kind)
            if args.only:
                chosen = tuple(x.strip().lower() for x in args.only.split(","))
                candidates = [x for x in candidates if any(x[0].lower() == pick or
                                                           x[0].lower().startswith(pick + "_")
                                                           for pick in chosen)]
            if not candidates:
                raise ValueError(f"No model matches --only for {pollutant}")
            for method, features in candidates:
                tr, va, te = split["train"], split["validation"], split["test"]
                xtr, ytr = tr[features].to_numpy(float), tr.target.to_numpy(float)
                xv, yv = va[features].to_numpy(float), va.target.to_numpy(float)
                if method.startswith("NN_"):
                    model, params = cal.build_neural(xtr, ytr, xv, yv,
                                                    method.split("_")[1], args.epochs, args.seed)
                else:
                    model, params = cal.build_linear(method, xtr, ytr, xv, yv)
                train_min = tr[features].quantile(0.01).to_dict()
                train_max = tr[features].quantile(0.99).to_dict()
                info = {
                    "sensor": sensor, "pollutant": pollutant, "kind": kind, "model": method,
                    "features": features, "parameters": params,
                    "calibration_train_days": "|".join(map(str, sorted(tr.day.unique()))),
                    "calibration_validation_days": "|".join(map(str, sorted(va.day.unique()))),
                    "calibration_test_days": "|".join(map(str, sorted(te.day.unique()))),
                    "train_quantile_01": train_min, "train_quantile_99": train_max,
                }
                for split_name, frame in split.items():
                    measured = frame.target.to_numpy(float)
                    predicted = model.predict(frame[features].to_numpy(float))
                    info[f"{split_name}_RMSE"] = cal.statistics(measured, predicted)["RMSE"]
                    info[f"{split_name}_R2"] = cal.statistics(measured, predicted)["R2"]
                fitted[(sensor, pollutant, method)] = {"estimator": model, **info}
                linear_formula = linear_coefficients(model, features)
                rows.append({k: v for k, v in info.items() if k not in ("train_quantile_01", "train_quantile_99")}
                            | {"linear_coefficients_json": json.dumps(linear_formula) if linear_formula else "",
                               "train_quantile_01_json": json.dumps(train_min),
                               "train_quantile_99_json": json.dumps(train_max)})
                print(f"[FIT] {sensor}/{pollutant}/{method}: validation RMSE={info['validation_RMSE']:.3f}")
    return fitted, pd.DataFrame(rows)


def linear_coefficients(model, features):
    if not hasattr(model, "named_steps"):
        return None
    scaling, linear = model.named_steps["standardize"], model.named_steps["model"]
    coefs = linear.coef_.reshape(-1) / scaling.scale_
    intercept = float(linear.intercept_ - np.dot(coefs, scaling.mean_))
    return {"intercept": intercept, "terms": {k: float(v) for k, v in zip(features, coefs)},
            "description": "Prediction = intercept + sum(terms[feature] * raw_feature). "
                           "Reference units and calibration instrument are unchanged."}


def transfer_session(event: dict, frame: pd.DataFrame, fitted: dict):
    sensor = event["sensor"]
    targets = ("CO", "NO2", "O3") if sensor == "gas_drone" else ("PM2_5", "PM10")
    result = {}
    stats_rows = []
    for pollutant in targets:
        related = [v for (dev, pol, _), v in fitted.items() if dev == sensor and pol == pollutant]
        if not related:
            continue
        base_raw = PRIMARY_RAW[pollutant]
        if base_raw not in frame:
            raise ValueError(f"{event['label']}: missing {base_raw}")
        info_cols = ["campaign", "session", "sensor", "source_file", "time_local", "meteo_source",
                     "gps_fix", "latitude", "longitude", "gps_alt_m", "gps_alt_relative_m", "speed_kmh",
                     "CO_raw", "NO2_raw", "O3_raw", "PM2_5_raw", "PM10_raw", "T", "RH", "p"]
        wide = frame.reindex(columns=[c for c in info_cols if c in frame]).copy()
        valid_raw = frame[base_raw].notna() & np.isfinite(frame[base_raw].to_numpy(float))
        n_raw = int(valid_raw.sum())
        if n_raw == 0:
            raise ValueError(f"{event['label']}: no valid raw measurements for {pollutant}")
        for info in related:
            features = info["features"]
            method = info["model"]
            # The PM fitting API calls the selected instrument signal PM_raw,
            # whereas flight exports retain PM2_5_raw and PM10_raw separately.
            inputs = frame.copy()
            if "PM_raw" in features:
                inputs["PM_raw"] = frame[base_raw]
            if not set(features).issubset(inputs.columns):
                missing = sorted(set(features) - set(inputs.columns))
                raise ValueError(f"{event['label']}/{method}: missing model features {missing}")
            x = inputs[features].copy().replace([np.inf, -np.inf], np.nan)
            valid = valid_raw & x.notna().all(axis=1)
            predicted = np.full(len(frame), np.nan)
            if valid.any():
                predicted[valid.to_numpy()] = info["estimator"].predict(x.loc[valid, features].to_numpy(float))
            finite = np.isfinite(predicted)
            negative = finite & (predicted < 0)
            wide[f"pred_{method}"] = predicted
            outside = pd.Series(False, index=x.index)
            for feat in features:
                outside |= (x[feat] < info["train_quantile_01"][feat]) | (x[feat] > info["train_quantile_99"][feat])
            n_pred = int(finite.sum())
            stats_rows.append({
                "campaign": event["campaign"], "session": event["session"], "sensor": sensor,
                "pollutant": pollutant, "model": method,
                "N_raw_seconds": n_raw, "N_predicted_seconds": n_pred,
                "coverage_pct": 100 * n_pred / n_raw,
                "N_negative": int(negative.sum()),
                "negative_pct": float(100 * negative.sum() / n_pred) if n_pred else np.nan,
                "minimum": float(np.min(predicted[finite])) if n_pred else np.nan,
                "median": float(np.median(predicted[finite])) if n_pred else np.nan,
                "maximum": float(np.max(predicted[finite])) if n_pred else np.nan,
                "p05": float(np.percentile(predicted[finite], 5)) if n_pred else np.nan,
                "p95": float(np.percentile(predicted[finite], 95)) if n_pred else np.nan,
                "outside_calibration_p01_p99_pct":
                    float(100 * (outside.to_numpy() & finite).sum() / n_pred) if n_pred else np.nan,
                "validation_RMSE": info["validation_RMSE"],
                "calibration_test_RMSE": info["test_RMSE"],
                "model_features": "|".join(features),
                "meteo_sources": "|".join(sorted(frame.loc[valid, "meteo_source"].dropna().unique())),
            })
        result[pollutant] = wide
    return result, stats_rows


def shortlist_models(stats: pd.DataFrame, fitted: dict, min_coverage: float):
    if stats.empty:
        raise ValueError("No transferred model predictions available")
    groupcols = ["sensor", "pollutant", "model"]
    grouped = []
    for keys, group in stats.groupby(groupcols):
        device, pollutant, method = keys
        worst_coverage = group.coverage_pct.min()
        nneg = int(group.N_negative.sum())
        n_sessions = len(group)
        n_camps = group.campaign.nunique()
        # Missing predictions never qualify as a pass for the negative-value gate.
        qualified = bool(nneg == 0 and group.N_predicted_seconds.min() > 0 and
                         worst_coverage >= 100 * min_coverage)
        causes = []
        if nneg:
            causes.append("negative predictions")
        if group.N_predicted_seconds.min() == 0:
            causes.append("no valid predictions in at least one session")
        if worst_coverage < 100 * min_coverage:
            causes.append("insufficient 1-second prediction coverage")
        grouped.append({
            "sensor": device, "pollutant": pollutant, "model": method,
            "shortlisted": qualified, "rejection_reason": "|".join(causes),
            "campaigns_checked": n_camps, "sessions_checked": n_sessions,
            "N_predicted_total": int(group.N_predicted_seconds.sum()),
            "N_negative_total": nneg,
            "negative_pct_total": float(100 * nneg / group.N_predicted_seconds.sum()),
            "minimum_predicted": group.minimum.min(),
            "minimum_session_coverage_pct": worst_coverage,
            "maximum_session_out_of_distribution_pct": group.outside_calibration_p01_p99_pct.max(),
            "validation_RMSE": group.validation_RMSE.iloc[0],
            "calibration_test_RMSE": group.calibration_test_RMSE.iloc[0],
        })
    out = pd.DataFrame(grouped).sort_values(
        ["sensor", "pollutant", "shortlisted", "validation_RMSE"], ascending=[True, True, False, True]
    )
    out["rank_among_shortlisted"] = np.nan
    for _, same in out.loc[out.shortlisted].groupby(["sensor", "pollutant"]):
        ordering = same.sort_values(["validation_RMSE", "calibration_test_RMSE"]).index
        out.loc[ordering, "rank_among_shortlisted"] = np.arange(1, len(ordering) + 1)
    return out


def save_csv(frame: pd.DataFrame, path: Path, with_index: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=";", decimal=",", index=with_index,
                 encoding="utf-8-sig", float_format="%.8g")


def plot_session(wide: pd.DataFrame, stats: pd.DataFrame, shortlist: pd.DataFrame,
                 event: dict, pollutant: str, output: Path, all_plots: bool):
    if wide.empty:
        return
    sensor = event["sensor"]
    candidates = shortlist.loc[(shortlist.sensor == sensor) & (shortlist.pollutant == pollutant)]
    selected = candidates.loc[candidates.shortlisted].sort_values("rank_among_shortlisted")
    names = selected.model.head(3).tolist()
    if not names:
        # Show three top validation models anyway, but make exclusion explicit.
        names = candidates.sort_values("validation_RMSE").model.head(3).tolist()
        suffix = "NO MODEL PASSED NEGATIVE-VALUE SCREEN"
    else:
        suffix = "Models passing strict screen"
    title = f"{CAMPAIGNS[event['campaign']]['name']} | {event['label']} | {pollutant}"
    time = wide.index
    plot_dir = output / "plots" / event["campaign"] / event["session"] / pollutant
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Time series plotted in 1-second resolution (no smoothing).
    fig, ax = plt.subplots(figsize=(12, 4.3))
    for name in names:
        col = f"pred_{name}"
        if col in wide:
            ax.plot(time, wide[col], lw=1.15, label=name)
    ax.axhline(0, color="black", ls="--", lw=0.9)
    ax.set(title=title + "\n1-second concentration time series | " + suffix,
           xlabel="Time (UTC)", ylabel="Predicted concentration (reference units)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    save_figure(fig, plot_dir / "concentration_vs_time_1s.png")

    # Altitude dependence from actual GPS fixes only; offset is relative, NOT height AGL.
    if "gps_alt_relative_m" in wide and wide.gps_alt_relative_m.notna().sum() > 10:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in names:
            v = wide[f"pred_{name}"]
            mask = v.notna() & wide.gps_alt_relative_m.notna()
            ax.scatter(v[mask], wide.loc[mask, "gps_alt_relative_m"],
                       alpha=0.15, s=10, label=name)
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set(title=title + "\nGPS-based exploratory vertical profile",
               xlabel="Predicted concentration (reference units)",
               ylabel="GPS altitude minus local 5th percentile (m; not AGL)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        save_figure(fig, plot_dir / "vertical_profile_1s.png")

    relevant = stats.loc[(stats.campaign == event["campaign"]) &
                         (stats.session == event["session"]) &
                         (stats.pollutant == pollutant)].sort_values("negative_pct")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(relevant) + 1.6)))
    ax.barh(relevant.model, relevant.negative_pct, color="tab:gray")
    ax.set(xlabel="Negative predictions (%)", title=title + "\nUnmodified model predictions")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    save_figure(fig, plot_dir / "negative_prediction_comparison.png")

    if all_plots:
        for name in candidates.model:
            fig, ax = plt.subplots(figsize=(11, 3.7))
            ax.plot(time, wide[f"pred_{name}"], linewidth=0.9)
            ax.axhline(0, color="tab:red", ls="--", lw=1)
            ax.set(title=title + f" | {name}", xlabel="Time (UTC)",
                   ylabel="Predicted concentration (reference units)")
            ax.grid(alpha=0.2)
            save_figure(fig, plot_dir / "individual_models" / f"{slug(name)}.png")


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def campaign_summary(series_cache, stats, shortlist, output):
    rows = []
    for (campaign, pollutant), parts in series_cache.items():
        for event, wide in parts:
            matched = shortlist.loc[(shortlist.sensor == event["sensor"]) &
                                    (shortlist.pollutant == pollutant) &
                                    (shortlist.shortlisted)]
            if not matched.empty:
                best = matched.sort_values("rank_among_shortlisted").iloc[0].model
                prediction = wide[f"pred_{best}"]
                rows.append({"campaign": campaign, "campaign_name": CAMPAIGNS[campaign]["name"],
                             "session": event["session"], "sensor": event["sensor"],
                             "pollutant": pollutant, "chosen_model": best,
                             "N_seconds": int(prediction.notna().sum()),
                             "median_concentration": float(prediction.median()),
                             "minimum_concentration": float(prediction.min()),
                             "maximum_concentration": float(prediction.max()),
                             "note": "Exploratory 1-second transfer; not an independent flight reference validation"})
            else:
                rows.append({"campaign": campaign, "campaign_name": CAMPAIGNS[campaign]["name"],
                             "session": event["session"], "sensor": event["sensor"],
                             "pollutant": pollutant, "chosen_model": "NONE PASSED",
                             "N_seconds": 0, "median_concentration": np.nan,
                             "minimum_concentration": np.nan,
                             "maximum_concentration": np.nan,
                             "note": "No model satisfied zero negatives and coverage threshold"})
    save_csv(pd.DataFrame(rows), output / "recommended_campaign_profiles.csv")


def main():
    parser = argparse.ArgumentParser(description="Exploratory 1-second flight calibration and negative-prediction screening")
    parser.add_argument("--flights-root", type=Path, default=DEFAULT_FLIGHTS_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--calibration-zip", type=Path, default=None,
                        help="Optional Mandria calibration ZIP (e.g. zip.zip)")
    parser.add_argument("--zip-22", type=Path, default=None)
    parser.add_argument("--zip-23", type=Path, default=None)
    parser.add_argument("--log-008", type=Path, default=None)
    parser.add_argument("--log-009", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pm-reference-csv", type=Path, default=None,
                        help="Optional original-XLS-derived daily PM reference CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None,
                        help="Comma-separated models/prefixes, e.g. LR,MLR,Ridge,NN")
    parser.add_argument("--pollutants", default="CO,NO2,O3,PM2_5,PM10")
    parser.add_argument("--min-coverage", type=float, default=MIN_SHORTLIST_COVERAGE,
                        help="Minimum per-session prediction coverage, 0 to 1")
    parser.add_argument("--plot-each-model", action="store_true")
    parser.add_argument("--check", action="store_true", help="Inspect flight and calibration input files")
    args = parser.parse_args()
    args.pollutants = {x.strip().upper() for x in args.pollutants.split(",")}
    if not args.pollutants.issubset(PRIMARY_RAW):
        raise ValueError("Unknown pollutants. Use CO,NO2,O3,PM2_5,PM10")
    if not 0 < args.min_coverage <= 1:
        raise ValueError("--min-coverage must be in (0,1]")
    sources = flight_sources(args)
    intervals = mission_definitions()
    for campaign, files in sources.items():
        print(f"[CAMPAIGN] {CAMPAIGNS[campaign]['name']}")
        for source in files:
            print("   ", source.provenance)
    if args.check:
        prepare_training(args)
        for campaign, files in sources.items():
            for f in files:
                h = f.read().columns.tolist()
                print(f"[COLUMNS] {campaign}/{Path(f.name).name}: {', '.join(h)}")
        print("Input inspection completed; no training or predictions were performed.")
        return
    output = args.output or (args.flights_root / "results_flight_transfer")
    output.mkdir(parents=True, exist_ok=True)
    print("[NOTE] One-second flight transfer is exploratory; no negative predictions are clipped.")
    fitted, model_table = prepare_training(args)
    if not fitted:
        raise ValueError("No calibration models were fitted")
    all_sessions = build_campaign_frames(sources, intervals)
    stats_rows = []
    series_cache = {}
    coverage_rows = []
    for campaign, events in all_sessions.items():
        for event, df in events:
            records, lines = transfer_session(event, df, fitted)
            for pollutant, wide in records.items():
                target = output / "series_1s" / campaign / event["session"]
                save_csv(wide, target / f"{event['sensor']}_{pollutant}.csv", with_index=True)
                series_cache.setdefault((campaign, pollutant), []).append((event, wide))
            stats_rows.extend(lines)
            coverage_rows.append({"campaign": campaign, "session": event["session"],
                                  "sensor": event["sensor"], "one_second_bins": len(df),
                                  "gps_fixed_seconds": int((df.get("gps_fix", pd.Series(dtype=float)) == 1).sum()),
                                  "meteo_ground_seconds": int((df.meteo_source == "ground_basic").sum()),
                                  "meteo_onboard_seconds": int((df.meteo_source == "onboard_gas_drone").sum()),
                                  "meteo_missing_seconds": int((df.meteo_source == "missing").sum()),
                                  "time_start_utc": str(df.index.min()), "time_end_utc": str(df.index.max()),
                                  "note": event["window_source"]})
    stats = pd.DataFrame(stats_rows)
    shortlist = shortlist_models(stats, fitted, args.min_coverage)
    save_csv(stats, output / "negative_predictions_by_session.csv")
    save_csv(shortlist, output / "all_models_screening.csv")
    save_csv(shortlist.loc[shortlist.shortlisted], output / "shortlist_zero_negatives.csv")
    save_csv(shortlist.loc[~shortlist.shortlisted], output / "excluded_models.csv")
    save_csv(model_table, output / "original_calibration_validation_and_test.csv")
    save_csv(pd.DataFrame(coverage_rows), output / "flight_data_coverage.csv")
    save_csv(pd.DataFrame(intervals).drop(columns="source_match", errors="ignore"), output / "flight_windows.csv")
    campaign_summary(series_cache, stats, shortlist, output)
    for (campaign, pollutant), parts in series_cache.items():
        for event, wide in parts:
            plot_session(wide, stats, shortlist, event, pollutant, output, args.plot_each_model)
    config = {
        "note": "Exploratory 1-second transfer: gas calibrated on 30-second means, PM on daily means",
        "calibration_instrument": "gas_drone for gases; opc_drone for PM",
        "transfer_resolution_seconds": 1, "negative_filter": "strict: any prediction <0 excludes model",
        "missing_prediction_filter": f">={args.min_coverage:.0%} coverage required per session",
        "rank_after_filter": "Mandria validation RMSE ascending; calibration test RMSE as tie-breaker",
        "flight_data_used_for_fitting": False,
        "pm_warning": "1-second PM concentrations are exploratory extrapolations from daily-mean models",
        "meteorology_warning": "OPC weather from time-matched ground_basic (fallback gas_drone), not onboard",
        "location_note": "LOG008 and LOG009 are Munich recordings during a visit by Graz researchers, not Graz flights",
        "altitude_warning": "GPS 5th-percentile relative altitude is not confirmed above-ground height",
        "original_calibration_days_only": True, "seed": args.seed,
        "epochs": args.epochs, "pollutants": sorted(args.pollutants),
    }
    (output / "methods_and_assumptions.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\n[RESULTS] Screening by sensor/pollutant:")
    for (s, p), v in shortlist.groupby(["sensor", "pollutant"]):
        passed = v.loc[v.shortlisted].sort_values("rank_among_shortlisted")
        print(f"  {s}/{p}: {len(passed)}/{len(v)} models passed (zero negatives + coverage)")
        if not passed.empty:
            top = passed.iloc[0]
            print(f"    Validation-best eligible: {top.model}, RMSE={top.validation_RMSE:.4g}")
        else:
            print("    No eligible model; no fallback or clipping applied.")
    print(f"[DONE] Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
