"""
IMU processing pipeline:
  1. Sensor fusion (Madgwick via imufusion, fallback: subtract running mean)
  2. Step detection (Butterworth LPF + peak detection)
  3. Activity classification (rule-based sliding window)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

try:
    import imufusion
    _IMUFUSION_AVAILABLE = True
except ImportError:
    _IMUFUSION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sensor fusion
# ---------------------------------------------------------------------------

def fuse_imu(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Return linear acceleration magnitude (gravity removed) array."""
    if _IMUFUSION_AVAILABLE:
        return _fuse_madgwick(ax, ay, az, gx, gy, gz, sample_rate)
    return _fuse_fallback(ax, ay, az)


def _fuse_madgwick(ax, ay, az, gx, gy, gz, sample_rate):
    ahrs = imufusion.Ahrs()
    offset = imufusion.Offset(int(sample_rate))
    settings = imufusion.Settings(
        imufusion.Convention.NWU,
        0.5,   # gain
        2000,  # gyroscope range (deg/s)
        10,    # acceleration rejection
        20,    # magnetic rejection (unused)
        int(5 * sample_rate),  # recovery trigger period
    )
    ahrs.settings = settings

    n = len(ax)
    linear_accel = np.zeros((n, 3))
    accel = np.column_stack([ax, ay, az])
    gyro_deg = np.column_stack([np.degrees(gx), np.degrees(gy), np.degrees(gz)])

    for i in range(n):
        gyro_deg[i] = offset.update(gyro_deg[i])
        ahrs.update_no_magnetometer(gyro_deg[i], accel[i], 1.0 / sample_rate)
        linear_accel[i] = ahrs.linear_acceleration

    vm = np.linalg.norm(linear_accel, axis=1)
    return vm


def _fuse_fallback(ax, ay, az):
    """Subtract running mean to approximate gravity removal, return VM."""
    window = 51  # ~0.5 s at 100 Hz
    accel = np.column_stack([ax, ay, az])
    running_mean = pd.DataFrame(accel).rolling(window, center=True, min_periods=1).mean().values
    linear = accel - running_mean
    return np.linalg.norm(linear, axis=1)


# ---------------------------------------------------------------------------
# Step detection
# ---------------------------------------------------------------------------

def detect_steps(
    vm: np.ndarray,
    sample_rate: float,
    threshold: float = 1.2,
    min_stride_s: float = 0.3,
) -> tuple[np.ndarray, float]:
    """
    Returns (peak_indices, cadence_spm).
    vm should be VM in g units.  threshold is in g.
    """
    nyq = 0.5 * sample_rate
    cutoff = min(5.0, nyq * 0.9)
    b, a = butter(4, cutoff / nyq, btype="low")
    filtered = filtfilt(b, a, vm)

    min_dist = int(min_stride_s * sample_rate)
    peaks, _ = find_peaks(filtered, height=threshold, distance=min_dist)

    cadence = 0.0
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / sample_rate  # seconds
        cadence = 60.0 / np.mean(intervals)

    return peaks, cadence


# ---------------------------------------------------------------------------
# Activity classification
# ---------------------------------------------------------------------------

ACTIVITIES = ["Sedentary", "Walking", "Running", "Cycling"]

_RULES = [
    # (vm_mean_min, vm_mean_max, vm_std_min, vm_std_max, freq_min, freq_max, label)
    (0.0,  1.05, 0.0,  0.05,  0.0,  0.5,  "Sedentary"),
    (0.9,  1.35, 0.04, 0.25,  1.0,  2.5,  "Walking"),
    (1.2,  99.,  0.15, 99.,   2.0,  5.0,  "Running"),
    (0.9,  1.25, 0.04, 0.18,  0.5,  1.5,  "Cycling"),
]


def classify_activity(
    vm_window: np.ndarray,
    sample_rate: float,
    accel_unit: str = "m/s2",
) -> str:
    """Classify a window of VM samples. accel_unit: 'm/s2' or 'g'."""
    if accel_unit == "m/s2":
        vm_g = vm_window / 9.81
    else:
        vm_g = vm_window.copy()

    mean_vm = float(np.mean(vm_g))
    std_vm = float(np.std(vm_g))

    fft_mag = np.abs(np.fft.rfft(vm_g - mean_vm))
    freqs = np.fft.rfftfreq(len(vm_g), d=1.0 / sample_rate)
    dominant_freq = float(freqs[np.argmax(fft_mag)]) if len(fft_mag) > 1 else 0.0

    for vm_lo, vm_hi, std_lo, std_hi, f_lo, f_hi, label in _RULES:
        if (vm_lo <= mean_vm < vm_hi and
                std_lo <= std_vm < std_hi and
                f_lo <= dominant_freq < f_hi):
            return label

    # fallback: nearest match by VM mean
    if mean_vm > 1.3:
        return "Running"
    if std_vm < 0.05:
        return "Sedentary"
    return "Walking"


# ---------------------------------------------------------------------------
# Full offline pipeline for CSV data
# ---------------------------------------------------------------------------

def process_dataframe(
    df: pd.DataFrame,
    sample_rate: float,
    accel_unit: str = "m/s2",
    gyro_unit: str = "rad/s",
    window_s: float = 2.0,
    step_threshold: float = 1.2,
) -> dict:
    """
    Run the complete pipeline on a DataFrame with columns:
    timestamp, ax, ay, az, gx, gy, gz

    Returns dict with keys:
      vm, filtered_vm, peaks, step_count, cadence,
      activity_labels, activity_times, epoch_df
    """
    ax = df["ax"].values.astype(float)
    ay = df["ay"].values.astype(float)
    az = df["az"].values.astype(float)
    gx = df["gx"].values.astype(float)
    gy = df["gy"].values.astype(float)
    gz = df["gz"].values.astype(float)

    # Convert gyro to rad/s if needed
    if gyro_unit == "deg/s":
        gx, gy, gz = np.radians(gx), np.radians(gy), np.radians(gz)

    # Convert accel to g for step detection / classification
    if accel_unit == "m/s2":
        ax_g, ay_g, az_g = ax / 9.81, ay / 9.81, az / 9.81
    else:
        ax_g, ay_g, az_g = ax.copy(), ay.copy(), az.copy()

    vm = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)

    # Low-pass filter
    nyq = 0.5 * sample_rate
    cutoff = min(5.0, nyq * 0.9)
    b, a = butter(4, cutoff / nyq, btype="low")
    filtered_vm = filtfilt(b, a, vm)

    peaks, cadence = detect_steps(vm, sample_rate, threshold=step_threshold)

    # Activity classification per window
    win_size = int(window_s * sample_rate)
    n = len(vm)
    activity_labels = []
    activity_times = []

    for start in range(0, n, win_size):
        end = min(start + win_size, n)
        window_vm = vm[start:end]
        label = classify_activity(window_vm, sample_rate, accel_unit="g")
        t = df["timestamp"].iloc[start] if "timestamp" in df.columns else start / sample_rate
        activity_labels.append(label)
        activity_times.append(float(t))

    # Epoch summary (per minute)
    epoch_records = []
    if "timestamp" in df.columns:
        ts = df["timestamp"].values
        t0 = ts[0]
        duration_s = ts[-1] - t0
        for minute in range(int(duration_s // 60) + 1):
            t_start = t0 + minute * 60
            t_end = t_start + 60
            mask = (ts >= t_start) & (ts < t_end)
            peak_mask = (ts[peaks] >= t_start) & (ts[peaks] < t_end) if len(peaks) else np.array([], dtype=bool)
            n_steps = int(np.sum(ts[peaks] >= t_start) * (ts[peaks] < t_end).sum() if len(peaks) else 0)
            n_steps = int(np.sum((ts[peaks] >= t_start) & (ts[peaks] < t_end))) if len(peaks) else 0
            acts = [activity_labels[i] for i, at in enumerate(activity_times)
                    if t_start <= at < t_end]
            dominant = max(set(acts), key=acts.count) if acts else "Unknown"
            epoch_records.append({
                "minute": minute + 1,
                "t_start": pd.Timestamp(t_start, unit="s"),
                "steps": n_steps,
                "activity": dominant,
                "vm_mean": float(np.mean(vm[mask])) if mask.sum() > 0 else 0.0,
            })

    epoch_df = pd.DataFrame(epoch_records)

    return {
        "vm": vm,
        "filtered_vm": filtered_vm,
        "peaks": peaks,
        "step_count": len(peaks),
        "cadence": cadence,
        "activity_labels": activity_labels,
        "activity_times": activity_times,
        "epoch_df": epoch_df,
    }
