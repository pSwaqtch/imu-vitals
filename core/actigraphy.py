"""
PyActigraphy wrapper.

Converts raw IMU step/VM data into epoch-level activity counts and computes:
  IS, IV, RA, L5, M10, Cosinor (acrophase + amplitude), sleep scoring.

Falls back to manual implementations if pyActigraphy is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    # Monkey-patch removed NumPy types required by pyActigraphy
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'object'): np.object = object
    if not hasattr(np, 'typeDict'): np.typeDict = np.sctypeDict
    import pyActigraphy
    _PYACTI_AVAILABLE = True
except ImportError:
    _PYACTI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Epoch builder
# ---------------------------------------------------------------------------

def build_epoch_series(
    timestamps: np.ndarray,
    vm: np.ndarray,
    epoch_s: int = 60,
) -> pd.Series:
    """
    Bin VM data into epoch_s-second bins, returning an activity count Series
    with DatetimeIndex.
    """
    t0 = pd.Timestamp(timestamps[0], unit="s")
    t1 = pd.Timestamp(timestamps[-1], unit="s")
    freq = f"{epoch_s}s"

    idx = pd.date_range(start=t0.floor(freq), end=t1.ceil(freq), freq=freq)
    counts = np.zeros(len(idx), dtype=float)

    for i, epoch_start in enumerate(idx):
        epoch_end = epoch_start + pd.Timedelta(seconds=epoch_s)
        ts_dt = pd.to_datetime(timestamps, unit="s")
        mask = (ts_dt >= epoch_start) & (ts_dt < epoch_end)
        if mask.sum() > 0:
            counts[i] = float(np.mean(np.abs(vm[mask])) * 1000)  # activity counts

    return pd.Series(counts, index=idx)


# ---------------------------------------------------------------------------
# Manual metric implementations (used as fallback)
# ---------------------------------------------------------------------------

def _IS(series: pd.Series) -> float:
    """Interdaily Stability: ratio of variance of average 24h profile to total variance."""
    if len(series) < 48:
        return float("nan")
    p = series.groupby(series.index.time).mean()
    overall_mean = series.mean()
    n = len(series)
    p_n = len(p)
    numerator = n / p_n * np.sum((p - overall_mean) ** 2)
    denominator = np.sum((series - overall_mean) ** 2)
    return float(numerator / denominator) if denominator != 0 else float("nan")


def _IV(series: pd.Series) -> float:
    """Intradaily Variability: ratio of mean squared diff to variance."""
    if len(series) < 2:
        return float("nan")
    diff_sq = np.diff(series.values) ** 2
    variance = np.var(series.values)
    return float(np.mean(diff_sq) / variance) if variance != 0 else float("nan")


def _RA(series: pd.Series) -> float:
    """Relative Amplitude: (M10 - L5) / (M10 + L5)."""
    l5, m10 = _L5(series), _M10(series)
    if np.isnan(l5) or np.isnan(m10):
        return float("nan")
    denom = m10 + l5
    return float((m10 - l5) / denom) if denom != 0 else float("nan")


def _rolling_mean_window(series: pd.Series, hours: float) -> tuple[float, pd.Timestamp]:
    """Return (mean_activity, start_time) for the consecutive window of `hours` hours
    with the given aggregation (min or max)."""
    n_epochs = int(hours * 3600 / _epoch_seconds(series))
    if n_epochs > len(series):
        return float("nan"), pd.Timestamp("NaT")
    rolled = series.rolling(n_epochs).mean()
    return float(rolled), pd.Timestamp("NaT")


def _epoch_seconds(series: pd.Series) -> int:
    if len(series) < 2:
        return 60
    return int((series.index[1] - series.index[0]).total_seconds())


def _L5(series: pd.Series) -> float:
    ep = _epoch_seconds(series)
    n = int(5 * 3600 / ep)
    if n > len(series):
        return float("nan")
    rolled = series.rolling(n).mean()
    return float(rolled.min())


def _M10(series: pd.Series) -> float:
    ep = _epoch_seconds(series)
    n = int(10 * 3600 / ep)
    if n > len(series):
        return float("nan")
    rolled = series.rolling(n).mean()
    return float(rolled.max())


def _cosinor(series: pd.Series) -> dict:
    """Fit a 24h cosine. Returns amplitude, acrophase (hours), mesor, r2."""
    if len(series) < 24:
        return {"amplitude": float("nan"), "acrophase_h": float("nan"),
                "mesor": float("nan"), "r2": float("nan")}
    t = np.array([(ts.hour + ts.minute / 60 + ts.second / 3600)
                  for ts in series.index])
    omega = 2 * np.pi / 24
    X = np.column_stack([np.ones(len(t)), np.cos(omega * t), np.sin(omega * t)])
    y = series.values
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        mesor, beta, gamma = coeffs
        amplitude = np.sqrt(beta**2 + gamma**2)
        acrophase_rad = np.arctan2(-gamma, beta)
        acrophase_h = (acrophase_rad / omega) % 24
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        return {"amplitude": float(amplitude), "acrophase_h": float(acrophase_h),
                "mesor": float(mesor), "r2": float(r2)}
    except Exception:
        return {"amplitude": float("nan"), "acrophase_h": float("nan"),
                "mesor": float("nan"), "r2": float("nan")}


# ---------------------------------------------------------------------------
# Cole-Kripke sleep scoring
# ---------------------------------------------------------------------------

_CK_WEIGHTS = [0.0, 0.0, 106.0, 54.0, 58.0, 76.0, 230.0, 74.0, 67.0]  # W(-4..+4)

def _cole_kripke_sleep(series: pd.Series) -> pd.Series:
    """
    Apply Cole-Kripke algorithm. Returns Series of 0 (sleep) / 1 (wake).
    Requires 1-min epochs.
    """
    values = series.values.copy()
    n = len(values)
    scores = np.zeros(n)
    p = 0.00001  # scale factor
    w = _CK_WEIGHTS

    for i in range(4, n - 4):
        D = p * (w[0] * values[i - 4] + w[1] * values[i - 3] +
                 w[2] * values[i - 2] + w[3] * values[i - 1] +
                 w[4] * values[i] +
                 w[5] * values[i + 1] + w[6] * values[i + 2] +
                 w[7] * values[i + 3] + w[8] * values[i + 4])
        scores[i] = 1 if D < 1.0 else 0  # 1=wake, 0=sleep

    return pd.Series(scores, index=series.index)


def _extract_sleep_periods(wake_series: pd.Series, min_sleep_min: int = 15) -> pd.DataFrame:
    """Extract contiguous sleep blocks from wake/sleep series."""
    sleep = (wake_series == 0).astype(int)
    records = []
    in_sleep = False
    start = None
    ep = _epoch_seconds(wake_series) // 60  # epochs per minute

    for ts, val in sleep.items():
        if val == 1 and not in_sleep:
            in_sleep = True
            start = ts
        elif val == 0 and in_sleep:
            in_sleep = False
            duration_min = int((ts - start).total_seconds() / 60)
            if duration_min >= min_sleep_min:
                records.append({"start": start, "end": ts, "duration_min": duration_min})
    if in_sleep and start is not None:
        duration_min = int((sleep.index[-1] - start).total_seconds() / 60)
        if duration_min >= min_sleep_min:
            records.append({"start": start, "end": sleep.index[-1], "duration_min": duration_min})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    timestamps: np.ndarray,
    vm: np.ndarray,
    epoch_s: int = 60,
) -> dict:
    """
    Main entry point. Returns dict with:
      series, IS, IV, RA, L5, M10, cosinor, sleep_df, duration_h
    """
    series = build_epoch_series(timestamps, vm, epoch_s)
    duration_h = float((timestamps[-1] - timestamps[0]) / 3600)

    if _PYACTI_AVAILABLE:
        metrics = _compute_with_pyactigraphy(series, duration_h)
    else:
        metrics = _compute_manual(series, duration_h)

    metrics["series"] = series
    metrics["duration_h"] = duration_h
    return metrics


def _compute_manual(series: pd.Series, duration_h: float) -> dict:
    is_val = _IS(series)
    iv_val = _IV(series)
    ra_val = _RA(series)
    l5_val = _L5(series)
    m10_val = _M10(series)
    cosinor = _cosinor(series)

    sleep_df = pd.DataFrame()
    if duration_h >= 24:
        wake_series = _cole_kripke_sleep(series)
        sleep_df = _extract_sleep_periods(wake_series)

    return {
        "IS": is_val,
        "IV": iv_val,
        "RA": ra_val,
        "L5": l5_val,
        "M10": m10_val,
        "cosinor": cosinor,
        "sleep_df": sleep_df,
    }


def _compute_with_pyactigraphy(series: pd.Series, duration_h: float) -> dict:
    """Use pyActigraphy if available; fall back to manual on any error."""
    try:
        from pyActigraphy.analysis import Cosinor
        raw = _make_raw(series)
        is_val = float(raw.IS())
        iv_val = float(raw.IV())
        ra_val = float(raw.RA())
        l5_val = float(raw.L5())
        m10_val = float(raw.M10())

        try:
            cosinor_obj = Cosinor(period="24h")
            cosinor_obj.fit(raw)
            cosinor = {
                "amplitude": float(cosinor_obj.amplitude),
                "acrophase_h": float(cosinor_obj.acrophase.total_seconds() / 3600),
                "mesor": float(cosinor_obj.mesor),
                "r2": float(cosinor_obj.r_squared),
            }
        except Exception:
            cosinor = _cosinor(series)

        sleep_df = pd.DataFrame()
        if duration_h >= 24:
            wake_series = _cole_kripke_sleep(series)
            sleep_df = _extract_sleep_periods(wake_series)

        return {
            "IS": is_val,
            "IV": iv_val,
            "RA": ra_val,
            "L5": l5_val,
            "M10": m10_val,
            "cosinor": cosinor,
            "sleep_df": sleep_df,
        }
    except Exception:
        return _compute_manual(series, duration_h)


def _make_raw(series: pd.Series):
    """Construct a minimal pyActigraphy-compatible raw object."""
    import pyActigraphy
    # Try generic BaseRaw construction path
    try:
        from pyActigraphy.io import BaseRaw
        raw = BaseRaw(
            name="imu_data",
            uuid="imu-0001",
            format="IMU",
            axial_mode=None,
            start_time=series.index[0],
            period=pd.Timedelta(series.index[-1] - series.index[0]),
            frequency=pd.Timedelta(series.index[1] - series.index[0]),
            data=series,
            light=None,
        )
        return raw
    except Exception:
        raise RuntimeError("Cannot construct pyActigraphy BaseRaw object")
