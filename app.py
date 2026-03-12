"""
Streamlit IMU Activity Monitor
Tabs: Live Stream | CSV Analysis | Circadian / Sleep
"""

from __future__ import annotations

import collections
import io
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import imu_processor as imp
from core import actigraphy as acti
from core import websocket_client as wsc

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper — activity timeline (defined early, used in multiple tabs)
# ---------------------------------------------------------------------------
def _plot_activity_timeline(
    hist: list[dict],
    key: str,
    t0: Optional[float] = None,
) -> None:
    if not hist:
        return
    color_map = {
        "Sedentary": "lightgray",
        "Walking": "steelblue",
        "Running": "tomato",
        "Cycling": "mediumseagreen",
        "Unknown": "white",
    }
    t_ref = t0 if t0 else hist[0]["t"]
    fig = go.Figure()
    bar_width = max((hist[1]["t"] - hist[0]["t"]) if len(hist) > 1 else 2, 0.5)
    for act in imp.ACTIVITIES + ["Unknown"]:
        segs = [h for h in hist if h["activity"] == act]
        if not segs:
            continue
        fig.add_trace(go.Bar(
            x=[h["t"] - t_ref for h in segs],
            y=[1] * len(segs),
            name=act,
            marker_color=color_map.get(act, "gray"),
            width=bar_width,
        ))
    fig.update_layout(
        title="Activity Timeline",
        barmode="stack",
        xaxis_title="Time (s)", yaxis=dict(visible=False),
        height=180, margin=dict(t=40, b=30),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


st.set_page_config(
    page_title="IMU Activity Monitor",
    page_icon="🏃",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    sample_rate = st.slider("Sample rate (Hz)", 10, 400, 100, step=10)
    accel_unit = st.selectbox("Accel units", ["m/s²", "g"])
    gyro_unit = st.selectbox("Gyro units", ["rad/s", "deg/s"])
    window_s = st.slider("Analysis window (s)", 1, 5, 2)
    step_thresh = st.slider("Step detection threshold (g)", 0.5, 3.0, 1.2, step=0.05)

    st.divider()
    st.caption("Sample rate affects filter cutoffs and step sensitivity.")

# Map display units to internal strings
_accel_unit_str = "m/s2" if "m" in accel_unit else "g"
_gyro_unit_str = "rad/s" if "rad" in gyro_unit else "deg/s"

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_csv, tab_circadian = st.tabs(
    ["📡 Live Stream", "📂 CSV Analysis", "🌙 Circadian / Sleep"]
)


# ===========================================================================
# TAB 1 — Live Stream
# ===========================================================================
with tab_live:
    st.header("Live IMU Stream")

    # Session-state initialisation
    if "imu_buffer" not in st.session_state:
        st.session_state["imu_buffer"] = collections.deque(maxlen=int(10 * sample_rate))
    if "session_steps" not in st.session_state:
        st.session_state["session_steps"] = 0
    if "session_start" not in st.session_state:
        st.session_state["session_start"] = None
    if "activity_history" not in st.session_state:
        st.session_state["activity_history"] = []

    ws_url = st.text_input(
        "WebSocket URL",
        value="ws://localhost:8765",
        placeholder="ws://192.168.1.100:8765",
    )

    col_conn, col_disc = st.columns(2)
    with col_conn:
        if st.button("🟢 Connect", use_container_width=True):
            if not wsc.get_client(st.session_state):
                wsc.start_client(st.session_state, ws_url)
                st.session_state["session_start"] = time.time()
                st.rerun()
    with col_disc:
        if st.button("🔴 Disconnect", use_container_width=True):
            wsc.stop_client(st.session_state)
            st.rerun()

    client = wsc.get_client(st.session_state)
    if client and client.is_running() and client.connected:
        st.success("Connected")
    elif client and client.error:
        st.error(f"WebSocket error: {client.error}")
        st.session_state["ws_error"] = client.error
    elif client and client.is_running():
        st.info("Connecting…")
    else:
        st.info("Not connected")

    # --- Live refresh fragment ---
    @st.fragment(run_every=0.2)
    def live_display():
        # Drain incoming samples
        samples = wsc.drain_queue(st.session_state)
        buf: collections.deque = st.session_state["imu_buffer"]
        buf.maxlen  # keep reference
        # Resize deque if sample_rate changed
        desired_len = int(10 * sample_rate)
        if buf.maxlen != desired_len:
            buf = collections.deque(buf, maxlen=desired_len)
            st.session_state["imu_buffer"] = buf

        for s in samples:
            buf.append(s)

        if len(buf) < 10:
            st.info("Waiting for data…")
            return

        data = list(buf)
        ts = np.array([d["t"] if d["t"] else 0.0 for d in data], dtype=float)
        ax = np.array([d["ax"] for d in data], dtype=float)
        ay = np.array([d["ay"] for d in data], dtype=float)
        az = np.array([d["az"] for d in data], dtype=float)
        vm_g = np.sqrt((ax / 9.81) ** 2 + (ay / 9.81) ** 2 + (az / 9.81) ** 2) \
               if _accel_unit_str == "m/s2" else np.sqrt(ax**2 + ay**2 + az**2)

        peaks, cadence = imp.detect_steps(vm_g, sample_rate, threshold=step_thresh)
        st.session_state["session_steps"] = len(peaks)

        # Activity on last window
        win = int(window_s * sample_rate)
        last_vm = vm_g[-win:] if len(vm_g) >= win else vm_g
        activity = imp.classify_activity(last_vm, sample_rate, accel_unit="g")

        hist = st.session_state["activity_history"]
        if ts[-1]:
            hist.append({"t": float(ts[-1]), "activity": activity})

        # Duration
        sess_start = st.session_state.get("session_start")
        duration_s = int(time.time() - sess_start) if sess_start else 0
        mm, ss = divmod(duration_s, 60)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", st.session_state["session_steps"])
        c2.metric("Cadence (spm)", f"{cadence:.0f}")
        c3.metric("Activity", activity)
        c4.metric("Duration", f"{mm:02d}:{ss:02d}")

        # Accelerometer chart
        t_axis = ts - ts[0] if ts[0] else np.arange(len(ts)) / sample_rate
        fig_acc = go.Figure()
        for label, vals in [("ax", ax), ("ay", ay), ("az", az)]:
            fig_acc.add_trace(go.Scatter(x=t_axis, y=vals, name=label, mode="lines"))
        fig_acc.add_trace(go.Scatter(
            x=t_axis, y=vm_g * (9.81 if _accel_unit_str == "m/s2" else 1),
            name="VM", mode="lines", line=dict(width=2, dash="dot"),
        ))
        fig_acc.update_layout(
            title="Accelerometer (rolling 10 s)",
            xaxis_title="Time (s)", yaxis_title=accel_unit,
            height=300, margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_acc, use_container_width=True, key="live_acc")

        # Activity timeline
        if hist:
            _plot_activity_timeline(hist, key="live_act")

    live_display()

    # Export button
    if len(st.session_state["imu_buffer"]) > 0:
        data = list(st.session_state["imu_buffer"])
        export_df = pd.DataFrame(data).rename(columns={"t": "timestamp"})
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Export session CSV",
            data=csv_bytes,
            file_name="imu_session.csv",
            mime="text/csv",
        )


# ===========================================================================
# TAB 2 — CSV Analysis
# ===========================================================================
with tab_csv:
    st.header("CSV Analysis")

    uploaded = st.file_uploader("Upload IMU CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = {"timestamp", "ax", "ay", "az", "gx", "gy", "gz"}
            missing = required - set(df.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.stop()

            with st.spinner("Processing…"):
                result = imp.process_dataframe(
                    df,
                    sample_rate=sample_rate,
                    accel_unit=_accel_unit_str,
                    gyro_unit=_gyro_unit_str,
                    window_s=window_s,
                    step_threshold=step_thresh,
                )

            vm = result["vm"]
            filtered_vm = result["filtered_vm"]
            peaks = result["peaks"]
            ts = df["timestamp"].values

            # Summary metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total steps", result["step_count"])
            c2.metric("Avg cadence (spm)", f"{result['cadence']:.0f}")
            duration_s = ts[-1] - ts[0]
            c3.metric("Duration", f"{int(duration_s//60)}m {int(duration_s%60)}s")

            # Accelerometer + peaks
            t_axis = ts - ts[0]
            fig_acc = go.Figure()
            for col, label in [("ax", "ax"), ("ay", "ay"), ("az", "az")]:
                fig_acc.add_trace(go.Scatter(
                    x=t_axis, y=df[col].values, name=label, mode="lines", opacity=0.7
                ))
            fig_acc.add_trace(go.Scatter(
                x=t_axis, y=vm, name="VM (g)", mode="lines",
                line=dict(color="black", width=1.5),
            ))
            fig_acc.add_trace(go.Scatter(
                x=t_axis, y=filtered_vm, name="Filtered VM",
                mode="lines", line=dict(dash="dash", color="orange"),
            ))
            if len(peaks):
                fig_acc.add_trace(go.Scatter(
                    x=t_axis[peaks], y=vm[peaks], name="Steps",
                    mode="markers", marker=dict(size=6, color="red", symbol="x"),
                ))
            fig_acc.update_layout(
                title="Accelerometer + Step Markers",
                xaxis_title="Time (s)", yaxis_title=accel_unit,
                height=350, margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # Activity timeline
            hist = [
                {"t": float(at), "activity": label}
                for at, label in zip(result["activity_times"], result["activity_labels"])
            ]
            _plot_activity_timeline(hist, key="csv_act", t0=float(ts[0]))

            # Epoch table
            st.subheader("Per-epoch summary")
            st.dataframe(result["epoch_df"], use_container_width=True)

            # Store for circadian tab
            st.session_state["csv_timestamps"] = ts
            st.session_state["csv_vm"] = vm

        except Exception as e:
            st.exception(e)
    else:
        st.info("Upload a CSV file to begin analysis.  "
                "A sample file is available at `sample_data/sample_imu.csv`.")


# ===========================================================================
# TAB 3 — Circadian / Sleep
# ===========================================================================
with tab_circadian:
    st.header("Circadian Rhythm & Sleep Analysis")

    source = st.radio(
        "Data source",
        ["Use live session data", "Upload separate CSV"],
        horizontal=True,
    )

    circ_ts: Optional[np.ndarray] = None
    circ_vm: Optional[np.ndarray] = None

    if source == "Use live session data":
        if "csv_timestamps" in st.session_state:
            circ_ts = st.session_state["csv_timestamps"]
            circ_vm = st.session_state["csv_vm"]
            st.success(f"Using CSV session data — {len(circ_ts)} samples")
        else:
            buf_data = list(st.session_state.get("imu_buffer", []))
            if buf_data:
                circ_ts = np.array([d.get("t") or 0.0 for d in buf_data])
                raw_ax = np.array([d["ax"] for d in buf_data])
                raw_ay = np.array([d["ay"] for d in buf_data])
                raw_az = np.array([d["az"] for d in buf_data])
                if _accel_unit_str == "m/s2":
                    circ_vm = np.sqrt((raw_ax / 9.81)**2 + (raw_ay / 9.81)**2 + (raw_az / 9.81)**2)
                else:
                    circ_vm = np.sqrt(raw_ax**2 + raw_ay**2 + raw_az**2)
                st.success(f"Using live buffer — {len(circ_ts)} samples")
            else:
                st.info("No live session data yet. Connect to WebSocket first, or upload a CSV.")

    else:  # Upload
        circ_upload = st.file_uploader("Upload CSV for circadian analysis", type=["csv"], key="circ_up")
        if circ_upload:
            circ_df = pd.read_csv(circ_upload)
            required = {"timestamp", "ax", "ay", "az"}
            if not required.issubset(circ_df.columns):
                st.error(f"Need at least: {required}")
            else:
                circ_ts = circ_df["timestamp"].values.astype(float)
                ax_c = circ_df["ax"].values.astype(float)
                ay_c = circ_df["ay"].values.astype(float)
                az_c = circ_df["az"].values.astype(float)
                if _accel_unit_str == "m/s2":
                    circ_vm = np.sqrt((ax_c / 9.81)**2 + (ay_c / 9.81)**2 + (az_c / 9.81)**2)
                else:
                    circ_vm = np.sqrt(ax_c**2 + ay_c**2 + az_c**2)

    if circ_ts is not None and circ_vm is not None and len(circ_ts) > 120:
        with st.spinner("Computing circadian metrics…"):
            metrics = acti.compute_metrics(circ_ts, circ_vm, epoch_s=60)

        duration_h = metrics["duration_h"]
        st.info(f"Data duration: **{duration_h:.1f} hours**  "
                f"({'sufficient for sleep scoring' if duration_h >= 24 else 'need ≥24h for sleep scoring'})")

        # Metric cards
        st.subheader("Non-parametric Circadian Rhythm Metrics")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("IS", f"{metrics['IS']:.3f}" if not np.isnan(metrics["IS"]) else "N/A",
                   help="Interdaily Stability (0–1). Higher = more consistent daily rhythm.")
        mc2.metric("IV", f"{metrics['IV']:.3f}" if not np.isnan(metrics["IV"]) else "N/A",
                   help="Intradaily Variability (0–2). Lower = less fragmented rhythm.")
        mc3.metric("RA", f"{metrics['RA']:.3f}" if not np.isnan(metrics["RA"]) else "N/A",
                   help="Relative Amplitude (0–1). Higher = stronger rest-activity contrast.")
        mc4.metric("L5 (counts)", f"{metrics['L5']:.1f}" if not np.isnan(metrics["L5"]) else "N/A",
                   help="Mean activity in the least active 5-hour window.")
        mc5.metric("M10 (counts)", f"{metrics['M10']:.1f}" if not np.isnan(metrics["M10"]) else "N/A",
                   help="Mean activity in the most active 10-hour window.")

        # Cosinor
        c = metrics["cosinor"]
        st.subheader("Cosinor Analysis")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Amplitude", f"{c['amplitude']:.2f}" if not np.isnan(c.get("amplitude", float("nan"))) else "N/A")
        cc2.metric("Acrophase", f"{c['acrophase_h']:.1f} h" if not np.isnan(c.get("acrophase_h", float("nan"))) else "N/A",
                   help="Peak activity time (hours from midnight).")
        cc3.metric("Mesor", f"{c['mesor']:.2f}" if not np.isnan(c.get("mesor", float("nan"))) else "N/A",
                   help="24h mean activity level.")
        cc4.metric("R²", f"{c['r2']:.3f}" if not np.isnan(c.get("r2", float("nan"))) else "N/A")

        # Actogram + cosinor overlay
        series = metrics["series"]
        fig_cos = go.Figure()
        fig_cos.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name="Activity", mode="lines", line=dict(color="steelblue"),
        ))
        if not np.isnan(c.get("amplitude", float("nan"))):
            t_fit = np.linspace(0, (series.index[-1] - series.index[0]).total_seconds() / 3600,
                                len(series))
            omega = 2 * np.pi / 24
            fit = (c["mesor"] + c["amplitude"] *
                   np.cos(omega * (t_fit - c["acrophase_h"])))
            fig_cos.add_trace(go.Scatter(
                x=series.index, y=fit,
                name="Cosinor fit", mode="lines",
                line=dict(color="red", dash="dash", width=2),
            ))
        fig_cos.update_layout(
            title="Actogram with Cosinor Fit",
            xaxis_title="Date / Time", yaxis_title="Activity counts",
            height=350, margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_cos, use_container_width=True)

        # Double-plot actogram (24 h × 2)
        if duration_h >= 24:
            st.subheader("Double-plotted Actogram")
            ep = int((series.index[1] - series.index[0]).total_seconds() / 60)  # minutes per epoch
            epochs_per_day = int(24 * 60 / ep)
            fig_dp = go.Figure()
            vals = series.values
            n_days = int(np.ceil(len(vals) / epochs_per_day))
            for day in range(n_days):
                seg = vals[day * epochs_per_day: (day + 1) * epochs_per_day]
                x = np.arange(len(seg)) * ep / 60  # hours
                # plot day in positions 0–24 and 24–48
                for offset in [0, 24]:
                    fig_dp.add_trace(go.Scatter(
                        x=x + offset, y=seg + day * (seg.max() + 5 if seg.max() > 0 else 10),
                        mode="lines", showlegend=False,
                        line=dict(color="navy", width=0.8),
                    ))
            fig_dp.update_layout(
                title="Double-plotted Actogram",
                xaxis_title="Hour of day", yaxis_title="Day (stacked)",
                xaxis=dict(tickvals=list(range(0, 49, 6))),
                height=max(300, 80 * n_days), margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_dp, use_container_width=True)

        # Sleep periods
        sleep_df = metrics.get("sleep_df", pd.DataFrame())
        if not sleep_df.empty:
            st.subheader("Detected Sleep Periods (Cole-Kripke)")
            st.dataframe(sleep_df, use_container_width=True)
        elif duration_h >= 24:
            st.info("No sleep periods detected (may need longer recording).")
        else:
            st.info("Upload ≥24 h of data to enable sleep scoring.")

    elif circ_ts is not None:
        st.warning("Need at least 2 minutes of data for circadian analysis.")


