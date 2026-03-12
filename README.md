# IMU Activity Monitor

A Streamlit app that ingests 6-axis IMU data (accelerometer + gyroscope) from a CSV file or a live WebSocket stream, performs real-time step counting and activity classification, and provides circadian rhythm / sleep analysis via PyActigraphy.

## Features

- **Live Stream** — connect to a hardware WebSocket, real-time step counting and activity display refreshed every 200 ms
- **CSV Analysis** — upload a recording and run the full processing pipeline offline
- **Circadian / Sleep** — compute IS, IV, RA, L5/M10, cosinor fit, and Cole-Kripke sleep scoring on long recordings

## Processing Pipeline

1. **Sensor fusion** — Madgwick AHRS via `imufusion` (falls back to running-mean gravity subtraction)
2. **Step detection** — Butterworth low-pass filter at 5 Hz + `scipy.signal.find_peaks`
3. **Activity classification** — rule-based sliding-window classifier (Sedentary / Walking / Running / Cycling) using VM mean, std, and dominant FFT frequency

## Data Formats

### CSV (upload or export)
```
timestamp,ax,ay,az,gx,gy,gz
1710000000.000,0.01,-0.02,9.81,0.001,0.002,-0.003
```
- `ax/ay/az` — accelerometer in m/s² or g (select in sidebar)
- `gx/gy/gz` — gyroscope in rad/s or deg/s (select in sidebar)

### WebSocket JSON (per sample)
```json
{"t": 1710000000.000, "ax": 0.01, "ay": -0.02, "az": 9.81,
 "gx": 0.001, "gy": 0.002, "gz": -0.003}
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install all deps
pip install streamlit numpy pandas scipy plotly imufusion websockets

# pyActigraphy has a broken transitive dep (scikit-learn==1.0.1) on Python >= 3.12
# Install without deps — the app falls back to built-in metric implementations
pip install pyActigraphy --no-deps

# Generate sample data (5-minute synthetic recording)
python generate_sample.py

# Launch
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## File Structure

```
step-counter/
├── app.py                    # Main Streamlit app (3 tabs + sidebar)
├── core/
│   ├── imu_processor.py      # Madgwick fusion, step detection, activity classifier
│   ├── actigraphy.py         # PyActigraphy wrapper: IS/IV/RA/L5/M10/Cosinor/sleep
│   └── websocket_client.py   # Background-thread async WebSocket client + queue
├── generate_sample.py        # Generates sample_data/sample_imu.csv
├── requirements.txt
└── sample_data/
    └── sample_imu.csv        # 5-min synthetic demo recording
```

## Sidebar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Sample rate | 100 Hz | Affects filter cutoffs and step detection |
| Accel units | m/s² | Switch to g if your device outputs g |
| Gyro units | rad/s | Switch to deg/s if needed |
| Analysis window | 2 s | Sliding window size for activity classification |
| Step threshold | 1.2 g | Peak height threshold for step detection |

## Demo Data

`sample_data/sample_imu.csv` — 5-minute synthetic recording at 100 Hz:

| Segment | Duration | Activity | Expected steps |
|---------|----------|----------|---------------|
| 0–60 s | 60 s | Sedentary | 0 |
| 60–180 s | 120 s | Walking (1.8 Hz) | ~216 |
| 180–240 s | 60 s | Running (3 Hz) | ~180 |
| 240–300 s | 60 s | Cycling (1 Hz) | — |

## Circadian Analysis

Requires accumulated data from the live session or an uploaded CSV:

- **≥ 2 hours** — IS, IV, RA, Cosinor fit
- **≥ 24 hours** — L5/M10, double-plotted actogram, Cole-Kripke sleep scoring

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | UI framework |
| numpy / pandas / scipy | Signal processing |
| plotly | Interactive charts |
| imufusion | Madgwick AHRS sensor fusion |
| websockets | Async WebSocket client |
| pyActigraphy | Circadian rhythm analysis (optional, manual fallback available) |
