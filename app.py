"""
Bearing Fault Vibration Simulator (Streamlit)  -  physics engine: bearing_sim.py

Run:   streamlit run app.py
Needs: bearing_sim.py (and optionally calibrated_cwru.json) in the same folder.
"""
import io
import json
import os
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import io as sio
from scipy import stats

from bearing_sim import Bearing, SimConfig, envelope_spectrum, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "calibrated_cwru.json")
MM_PER_IN = 25.4
COLORS = {"outer": "tab:red", "inner": "tab:green", "ball": "tab:purple"}

PRESETS = {
    "CWRU 6205-2RS (drive end)": dict(n=9, D=39.04, d=7.94, beta=0.0, fs=12000, rpm=1797),
    "CWRU 6203-2RS (fan end)": dict(n=9, D=28.50, d=6.75, beta=0.0, fs=12000, rpm=1797),
    "Custom": dict(n=7, D=60.0, d=20.0, beta=0.0, fs=12000, rpm=1800),
}


# ============================================================ engine helpers
def cfg_to_json(cfg: SimConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True, default=list)


def cfg_from_dict(d: dict) -> SimConfig:
    d = dict(d)
    tup = lambda m: tuple(tuple(x) for x in m)
    d["modes"] = tup(d["modes"])
    d["fault_modes"] = {k: tup(v) for k, v in d.get("fault_modes", {}).items()}
    d["shaft_harm_amp"] = tuple(d["shaft_harm_amp"])
    return SimConfig(**d)


def load_calibrated():
    if not os.path.exists(CALIB_PATH):
        return None
    with open(CALIB_PATH) as fh:
        return cfg_from_dict(json.load(fh))


def with_size(cfg: SimConfig, fault: str, size_in: float) -> SimConfig:
    """Calibrated per-case amplitudes: interpolate between the measured sizes
    (clamped at the ends) so any size in mm can be requested."""
    pts = sorted((float(k.split("@")[1]), v) for k, v in cfg.case_gain.items()
                 if k.startswith(fault + "@"))
    if not pts:
        return cfg
    s, a = zip(*pts)
    amp = float(np.interp(size_in, s, a))
    return replace(cfg, case_gain={**cfg.case_gain, f"{fault}@{size_in:.3f}": amp})


def generate(bearing, faults, size_mm, rpm, cfg, load_hp, seed):
    """Background (noise + shaft harmonics) added ONCE; fault components summed."""
    size_in = size_mm / MM_PER_IN
    x, _ = simulate(bearing, "normal", 0.0, rpm, cfg, load_hp, seed)
    quiet = replace(cfg, noise_std=0.0, coloured_noise_std=0.0, shaft_harm_amp=(0, 0, 0))
    comps = {}
    for i, f in enumerate(faults):
        xf, _ = simulate(bearing, f, size_in, rpm, with_size(quiet, f, size_in),
                         load_hp, seed + 1 + i)
        comps[f] = xf
        x = x + xf
    return x, comps


@st.cache_data(show_spinner="Simulating ...")
def cached_generate(bearing_t, faults_t, size_mm, rpm, cfg_json, load_hp, seed):
    return generate(Bearing(*bearing_t), list(faults_t), size_mm, rpm,
                    cfg_from_dict(json.loads(cfg_json)), load_hp, seed)


def fault_freq(freqs, fault):
    # ball faults strike both races per ball rotation -> impact rate is 2*BSF
    return {"outer": freqs["BPFO"], "inner": freqs["BPFI"], "ball": 2 * freqs["BSF"]}[fault]


def env_peak_snr(f, s, f0, tol=0.02):
    floor = np.median(s[(f > 5) & (f < max(600.0, 6 * f0))])
    m = (f > f0 * (1 - tol)) & (f < f0 * (1 + tol))
    return float(s[m].max() / floor) if m.any() else float("nan")


def features(x):
    x = np.asarray(x, float) - np.mean(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {"RMS": rms, "Kurtosis": float(stats.kurtosis(x, fisher=False)),
            "Crest factor": float(np.max(np.abs(x)) / rms), "Skewness": float(stats.skew(x)),
            "Peak-to-peak": float(np.ptp(x))}


def amp_spectrum(x, fs):
    w = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / fs), np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()


def mark_harmonics(ax, items, xmax, nh=5):
    for name, f0, color in items:
        for k in range(1, nh + 1):
            if k * f0 < xmax:
                ax.axvline(k * f0, color=color, ls=":", lw=1,
                           label=f"{name} {f0:.1f} Hz (+harmonics)" if k == 1 else None)


def read_uploaded(file):
    """Return dict name -> 1-D array from a .mat or .csv upload."""
    if file.name.lower().endswith(".mat"):
        m = sio.loadmat(io.BytesIO(file.getvalue()))
        out = {k: np.squeeze(v).astype(float) for k, v in m.items()
               if not k.startswith("__") and np.squeeze(v).ndim == 1 and np.squeeze(v).size > 1000}
        return out
    df = pd.read_csv(io.BytesIO(file.getvalue()))
    return {c: df[c].to_numpy(float) for c in df.select_dtypes("number").columns}


# ================================================================= UI: setup
st.set_page_config(page_title="Bearing Fault Simulator", layout="wide")
st.title("Bearing Fault Vibration Simulator (physics-based, CWRU-calibratable)")

# ----------------------------------------------------------- bearing geometry
st.sidebar.header("Bearing geometry")
preset = st.sidebar.selectbox("Preset", list(PRESETS))
P = PRESETS[preset]
n_balls = st.sidebar.number_input("Number of rolling elements (n)", min_value=3, max_value=40,
                                  value=P["n"], key=f"n_{preset}")
D_mm = st.sidebar.number_input("Pitch diameter D [mm]", min_value=1.0, value=P["D"],
                               key=f"D_{preset}")
d_mm = st.sidebar.number_input("Ball diameter d [mm]", min_value=0.5, value=P["d"],
                               key=f"d_{preset}")
beta_deg = st.sidebar.number_input("Contact angle [deg]", min_value=0.0, max_value=60.0,
                                   value=P["beta"], key=f"b_{preset}")
fs = int(st.sidebar.number_input("Sampling frequency [Hz]", min_value=2000, max_value=100000,
                                 value=P["fs"], step=1000, key=f"fs_{preset}"))
duration = st.sidebar.number_input("Signal duration [s]", min_value=0.5, max_value=60.0, value=5.0)
rpm = st.sidebar.number_input("Shaft speed [RPM]", min_value=100.0, max_value=10000.0,
                              value=float(P["rpm"]), key=f"rpm_{preset}")
load_hp = st.sidebar.slider("Load [hp] (amplitude grows ~4 %/hp, uncalibrated)", 0.0, 3.0, 0.0, 1.0)
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=0))

# ----------------------------------------------------------------- faults
st.sidebar.header("Fault")
fault_types = st.sidebar.multiselect("Fault types (empty = healthy)", ["outer", "inner", "ball"],
                                     default=["outer"])
fault_size_mm = st.sidebar.number_input("Fault size [mm]", min_value=0.05, max_value=3.0,
                                        value=0.5334, step=0.0254, format="%.4f")
st.sidebar.caption(f"= {fault_size_mm / MM_PER_IN:.4f} in   (CWRU: 0.1778 / 0.3556 / 0.5334 / 0.7112 mm)")

# ----------------------------------------------------- realism / calibration
st.sidebar.header("Signal realism")
calib = load_calibrated()
use_calib = False
if calib is not None:
    use_calib = st.sidebar.checkbox("Use CWRU-calibrated parameters (calibrated_cwru.json)", False)
    if use_calib:
        st.sidebar.caption("Resonances, noise, slip and per-case amplitudes come from the real CWRU "
                           "fit (amplitudes in sensor units, interpolated between 0.18/0.36/0.53 mm). "
                           "Manual noise/resonance settings below are ignored.")
mod_depth = st.sidebar.slider("Load-zone modulation depth (inner: 1x shaft, ball: cage)", 0.0, 1.0, 0.8)
noise_std = st.sidebar.slider("White noise std", 0.0, 0.5, 0.05)
coloured_std = st.sidebar.slider("Coloured (low-frequency) noise std", 0.0, 0.3, 0.03)

with st.sidebar.expander("Advanced physics"):
    jitter = st.slider("Slip / jitter (fraction of impact period)", 0.0, 0.15, 0.01, 0.005)
    scatter = st.slider("Impact amplitude scatter", 0.0, 0.6, 0.15, 0.05)
    speed_fluct = st.slider("Speed fluctuation", 0.0, 0.03, 0.005, 0.001)
    size_exp = st.slider("Amplitude vs size exponent", 0.0, 2.0, 0.8, 0.1)
    a0 = st.number_input("Amplitude gain (a0)", min_value=0.0, value=1.0)
    h1 = st.number_input("Shaft 1x amplitude", min_value=0.0, value=0.05)
    h2 = st.number_input("Shaft 2x amplitude", min_value=0.0, value=0.02)
    h3 = st.number_input("Shaft 3x amplitude", min_value=0.0, value=0.01)
    st.caption("Structural resonances (impact response)")
    modes_df = st.data_editor(
        pd.DataFrame(SimConfig().modes, columns=["freq_Hz", "damping", "gain"]),
        num_rows="dynamic", key="modes_editor")

st.sidebar.header("Visualization")
zoom_ms = st.sidebar.slider("Time-plot window [ms]", 20, 1000, 200)
lo_def, hi_def = int(fs * 0.1667), int(fs * 0.4167)
band = st.sidebar.slider("Envelope demodulation band [Hz]", 200, int(fs * 0.48), (lo_def, hi_def))
show_harmonics = st.sidebar.checkbox("Mark fault harmonics", True)
show_envelope = st.sidebar.checkbox("Show envelope spectrum", True)

# ------------------------------------------------------------ build config
bearing = Bearing("custom", int(n_balls), float(d_mm), float(D_mm), float(beta_deg))
if d_mm >= D_mm:
    st.error("Ball diameter must be smaller than pitch diameter.")
    st.stop()

if use_calib:
    cfg = replace(calib, fs=fs, duration=float(duration),
                  inner_mod_depth=mod_depth, ball_mod_depth=mod_depth)
    if fs != 12000 or not preset.startswith("CWRU 6205"):
        st.warning("Calibration is valid for CWRU 6205 @ 12 kHz only.")
else:
    modes = tuple((float(r.freq_Hz), float(r.damping), float(r.gain))
                  for r in modes_df.dropna().itertuples() if 0 < r.freq_Hz < 0.48 * fs)
    if not modes:
        st.error("No resonance below 0.48*fs. Edit the resonance table.")
        st.stop()
    cfg = SimConfig(fs=fs, duration=float(duration), modes=modes, a0=a0,
                    size_exponent=size_exp, jitter=jitter, ir_amp_scatter=scatter,
                    speed_fluct=speed_fluct, noise_std=noise_std,
                    coloured_noise_std=coloured_std, shaft_harm_amp=(h1, h2, h3),
                    inner_mod_depth=mod_depth, ball_mod_depth=mod_depth)

# ----------------------------------------------------------- frequencies
fr = rpm / 60.0
freqs = bearing.freqs(fr)
cols = st.columns(6)
for c, (k, v) in zip(cols, [("Shaft 1x", fr), ("BPFO", freqs["BPFO"]), ("BPFI", freqs["BPFI"]),
                            ("BSF", freqs["BSF"]), ("2xBSF (ball impacts)", 2 * freqs["BSF"]),
                            ("FTF", freqs["FTF"])]):
    c.metric(k, f"{v:.2f} Hz")

# ----------------------------------------------------------- simulate
bt = (bearing.name, bearing.n_balls, bearing.d, bearing.D, bearing.contact_deg)
x, comps = cached_generate(bt, tuple(fault_types), float(fault_size_mm), float(rpm),
                           cfg_to_json(cfg), float(load_hp), seed)
t = np.arange(len(x)) / fs
items = [(f, fault_freq(freqs, f), COLORS[f]) for f in fault_types]
env_f, env_s = envelope_spectrum(x, fs, band)
env_xmax = float(min(1500.0, 6 * max([f0 for _, f0, _ in items], default=100.0)))

# ----------------------------------------------------------- optional real data
real = None
with st.expander("Compare with a real recording (.mat or .csv)"):
    up = st.file_uploader("Upload CWRU .mat or a CSV with a vibration column", type=["mat", "csv"])
    if up is not None:
        arrays = read_uploaded(up)
        if arrays:
            names = list(arrays)
            default = next((i for i, k in enumerate(names) if k.endswith("DE_time")), 0)
            key = st.selectbox("Channel", names, index=default)
            real = arrays[key][: len(x)]
            st.caption(f"{len(real)/fs:.1f} s used, assuming the recording was sampled at {fs} Hz.")
        else:
            st.warning("No 1-D numeric channel found in the file.")

# ----------------------------------------------------------- tabs
tabs = st.tabs(["Time", "Spectrum", "Envelope", "Metrics", "Dataset generator"])

with tabs[0]:
    nz = int(zoom_ms * 1e-3 * fs)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t[:nz], x[:nz], lw=0.7, label="Total signal")
    if len(comps) > 1:
        for f, xf in comps.items():
            ax.plot(t[:nz], xf[:nz], lw=0.5, alpha=0.6, color=COLORS[f], label=f"{f} component")
    ax.set(xlabel="Time [s]", ylabel="Amplitude", title=f"Simulated signal - fault size {fault_size_mm} mm")
    ax.grid(True); ax.legend(loc="upper right")
    st.pyplot(fig); plt.close(fig)

with tabs[1]:
    f_, s_ = amp_spectrum(x, fs)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(f_, s_, lw=0.6, label="Simulated")
    if real is not None:
        fr_, sr_ = amp_spectrum(real, fs)
        ax.plot(fr_, sr_, lw=0.5, alpha=0.7, label="Real")
    ax.axvspan(*band, color="orange", alpha=0.12, label="demod band")
    if show_harmonics:
        mark_harmonics(ax, items, fs / 2)
    ax.set(xlabel="Frequency [Hz]", ylabel="Amplitude", title="Amplitude spectrum")
    ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

with tabs[2]:
    if show_envelope:
        fig, ax = plt.subplots(figsize=(14, 4))
        floor = np.median(env_s[(env_f > 5) & (env_f < 600)])
        ax.plot(env_f, env_s / floor, lw=0.8, label="Simulated")
        if real is not None:
            rf, rs = envelope_spectrum(real, fs, band)
            ax.plot(rf, rs / np.median(rs[(rf > 5) & (rf < 600)]), lw=0.8, alpha=0.7, label="Real")
        if show_harmonics:
            mark_harmonics(ax, items, env_xmax)
        ax.set(xlim=(0, env_xmax), xlabel="Frequency [Hz]", ylabel="Amplitude / noise floor",
               title=f"Envelope spectrum (band {band[0]}-{band[1]} Hz)")
        ax.grid(True); ax.legend()
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Enable 'Show envelope spectrum' in the sidebar.")

with tabs[3]:
    rows = {"Simulated": features(x)}
    if real is not None:
        rows["Real"] = features(real)
    st.dataframe(pd.DataFrame(rows).T.round(4))
    if items:
        pk = []
        for name, f0, _ in items:
            r = {"fault": name, "freq [Hz]": f0, "1x peak / floor": env_peak_snr(env_f, env_s, f0),
                 "2x peak / floor": env_peak_snr(env_f, env_s, 2 * f0)}
            if real is not None:
                rf, rs = envelope_spectrum(real, fs, band)
                r["1x (real)"] = env_peak_snr(rf, rs, f0)
                r["2x (real)"] = env_peak_snr(rf, rs, 2 * f0)
            pk.append(r)
        st.write("Envelope-spectrum fault peaks (ratio to local noise floor)")
        st.dataframe(pd.DataFrame(pk).set_index("fault").round(1))
    st.caption("Kurtosis/crest vary a lot between real CWRU files of the same fault; compare "
               "distributions over many segments rather than single files.")

with tabs[4]:
    st.write("Generate a labelled set of fixed-length windows (uses the current sidebar physics).")
    c1, c2, c3 = st.columns(3)
    b_faults = c1.multiselect("Classes", ["normal", "outer", "inner", "ball"],
                              default=["normal", "outer", "inner", "ball"])
    b_sizes = c2.text_input("Fault sizes [mm]", "0.1778, 0.3556, 0.5334")
    b_segs = int(c3.number_input("Windows per class/size", min_value=1, value=50))
    b_win = int(c3.number_input("Window length [samples]", min_value=256, value=2048, step=256))
    if st.button("Generate dataset"):
        try:
            sizes = [float(s) for s in b_sizes.split(",") if s.strip()]
        except ValueError:
            sizes = []
            st.error("Could not parse the fault sizes.")
        combos = [(f, s) for f in b_faults for s in ([0.0] if f == "normal" else sizes)]
        if combos:
            Xs, lab, sz = [], [], []
            bar = st.progress(0.0)
            for i, (f, s) in enumerate(combos):
                dur = b_segs * b_win / fs
                cfg_b = replace(cfg, duration=dur)
                xb, _ = generate(bearing, [] if f == "normal" else [f], s, rpm, cfg_b,
                                 load_hp, seed + 1000 * (i + 1))
                Xs.append(xb[: b_segs * b_win].reshape(b_segs, b_win))
                lab += [f] * b_segs
                sz += [s] * b_segs
                bar.progress((i + 1) / len(combos))
            buf = io.BytesIO()
            np.savez_compressed(buf, X=np.concatenate(Xs).astype(np.float32), fault=np.array(lab),
                                size_mm=np.array(sz), rpm=rpm, fs=fs)
            st.session_state["batch_npz"] = buf.getvalue()
            st.session_state["batch_info"] = f"{len(lab)} windows x {b_win} samples, {len(combos)} class/size combos"
    if "batch_npz" in st.session_state:
        st.success(st.session_state["batch_info"])
        st.download_button("Download dataset (.npz: X, fault, size_mm, rpm, fs)",
                           st.session_state["batch_npz"], "bearing_synthetic_dataset.npz")

# ----------------------------------------------------------- downloads
df = pd.DataFrame({"Time": t, "Signal": x})
st.download_button("Download signal CSV", df.to_csv(index=False, float_format="%.6g").encode(),
                   "vibration_signal.csv", "text/csv")
