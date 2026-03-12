"""Generate sample_data/sample_imu.csv — run once to create demo data."""
import numpy as np
import pandas as pd
import os

SAMPLE_RATE = 100  # Hz
DURATION_S = 300  # 5 minutes
N = SAMPLE_RATE * DURATION_S
t = np.linspace(0, DURATION_S, N, endpoint=False)
timestamps = 1710000000.0 + t

rng = np.random.default_rng(42)

ax = np.zeros(N)
ay = np.zeros(N)
az = np.ones(N) * 9.81  # gravity

# 0–60s: sedentary — tiny noise
mask_sed = t < 60
ax[mask_sed] = rng.normal(0, 0.05, mask_sed.sum())
ay[mask_sed] = rng.normal(0, 0.05, mask_sed.sum())
az[mask_sed] = 9.81 + rng.normal(0, 0.05, mask_sed.sum())

# 60–180s: walking — 1.8 Hz sinusoidal vertical + noise
mask_walk = (t >= 60) & (t < 180)
tw = t[mask_walk]
ax[mask_walk] = 0.5 * np.sin(2 * np.pi * 1.8 * tw) + rng.normal(0, 0.05, mask_walk.sum())
ay[mask_walk] = 0.2 * np.sin(2 * np.pi * 1.8 * tw + 0.5) + rng.normal(0, 0.05, mask_walk.sum())
az[mask_walk] = 9.81 + 1.2 * np.sin(2 * np.pi * 1.8 * tw) + rng.normal(0, 0.1, mask_walk.sum())

# 180–240s: running — 3 Hz, higher amplitude
mask_run = (t >= 180) & (t < 240)
tr = t[mask_run]
ax[mask_run] = 1.5 * np.sin(2 * np.pi * 3.0 * tr) + rng.normal(0, 0.15, mask_run.sum())
ay[mask_run] = 0.8 * np.sin(2 * np.pi * 3.0 * tr + 0.3) + rng.normal(0, 0.1, mask_run.sum())
az[mask_run] = 9.81 + 3.0 * np.sin(2 * np.pi * 3.0 * tr) + rng.normal(0, 0.2, mask_run.sum())

# 240–300s: cycling — 1 Hz smooth
mask_cyc = t >= 240
tc = t[mask_cyc]
ax[mask_cyc] = 0.3 * np.sin(2 * np.pi * 1.0 * tc) + rng.normal(0, 0.03, mask_cyc.sum())
ay[mask_cyc] = 0.6 * np.sin(2 * np.pi * 1.0 * tc + 1.0) + rng.normal(0, 0.03, mask_cyc.sum())
az[mask_cyc] = 9.81 + 0.5 * np.sin(2 * np.pi * 1.0 * tc) + rng.normal(0, 0.05, mask_cyc.sum())

gx = rng.normal(0, 0.01, N)
gy = rng.normal(0, 0.01, N)
gz = rng.normal(0, 0.01, N)

df = pd.DataFrame({
    "timestamp": np.round(timestamps, 3),
    "ax": np.round(ax, 4),
    "ay": np.round(ay, 4),
    "az": np.round(az, 4),
    "gx": np.round(gx, 5),
    "gy": np.round(gy, 5),
    "gz": np.round(gz, 5),
})

os.makedirs("sample_data", exist_ok=True)
df.to_csv("sample_data/sample_imu.csv", index=False)
print(f"Written {len(df)} rows to sample_data/sample_imu.csv")
