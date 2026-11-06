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

from bearing_sim import Bearing, SimConfig, config_from_dict, envelope_spectrum, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILES = {"cwru": os.path.join(HERE, "calibrated_cwru.json"),
               "sq": os.path.join(HERE, "calibrated_spectraquest.json")}
SQ_SEV_IN = {"L": 0.001, "M": 0.002, "H": 0.003}   # severity index/1000 (no physical size known)
MM_PER_IN = 25.4
COLORS = {"outer": "tab:red", "inner": "tab:green", "ball": "tab:purple"}

PRESETS = {
    "CWRU 6205-2RS (drive end)": dict(n=9, D=39.04, d=7.94, beta=0.0, fs=12000, rpm=1797, band=(2000, 5000)),
    "CWRU 6203-2RS (fan end)": dict(n=9, D=28.50, d=6.75, beta=0.0, fs=12000, rpm=1797, band=(2000, 5000)),
    "SpectraQuest MFS - Rexnord ER-12K": dict(n=8, D=31.75, d=7.94, beta=0.0, fs=25600, rpm=1200,
                                              band=(9000, 12000)),
    "Custom": dict(n=7, D=60.0, d=20.0, beta=0.0, fs=12000, rpm=1800, band=(2000, 5000)),
}


# ============================================================ engine helpers
def cfg_to_json(cfg: SimConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True, default=list)


cfg_from_dict = config_from_dict


def load_calibrated(kind):
    """{label: SimConfig}. CWRU: one config. SpectraQuest: one per accelerometer axis."""
    path = CALIB_FILES.get(kind)
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        d = json.load(fh)
    if kind == "cwru":
        return {"": cfg_from_dict(d)}
    return {k: cfg_from_dict(v) for k, v in d.items() if not k.startswith("_")}


_NUM_CASE_FIELDS = ("case_gfrac", "case_drift", "case_bg", "case_jitter", "case_scatter",
                    "case_kappa", "case_mod", "case_drift_tau")


def with_size(cfg: SimConfig, fault: str, size_in: float) -> SimConfig:
    """Calibrated per-case values: interpolate between the measured sizes (clamped at the ends) so any
    size in mm can be requested. Resonance modes use the nearest measured size."""
    pts = sorted((float(k.split("@")[1]), v) for k, v in cfg.case_gain.items()
                 if k.startswith(fault + "@"))
    if not pts:
        return cfg
    s, a = zip(*pts)
    key = f"{fault}@{size_in:.3f}"
    out = replace(cfg, case_gain={**cfg.case_gain, key: float(np.interp(size_in, s, a))})
    for fld in _NUM_CASE_FIELDS:
        d = getattr(cfg, fld)
        v = [(sz, d[f"{fault}@{sz:.3f}"]) for sz in s if f"{fault}@{sz:.3f}" in d]
        if v:
            sv, vv = zip(*v)
            out = replace(out, **{fld: {**d, key: float(np.interp(size_in, sv, vv))}})
    near = min(s, key=lambda z: abs(z - size_in))
    nk = f"{fault}@{near:.3f}"
    if nk in cfg.case_modes:
        out = replace(out, case_modes={**cfg.case_modes, key: cfg.case_modes[nk]})
    eqs = [(sz, cfg.case_eq[f"{fault}@{sz:.3f}"]) for sz in s if f"{fault}@{sz:.3f}" in cfg.case_eq]
    if eqs:
        grid = eqs[0][1][0]
        db = np.array([np.asarray(e[1][1]) for e in eqs])
        sv = [e[0] for e in eqs]
        row = [float(np.interp(size_in, sv, db[:, j])) for j in range(db.shape[1])]
        out = replace(out, case_eq={**cfg.case_eq, key: (tuple(grid), tuple(round(v, 2) for v in row))})
    return out


def generate(bearing, faults, size_mm, rpm, cfg, load_hp, seed):
    """One fault (or none): the calibrated generator exactly as fitted. Several faults: the fault
    components are summed on one shared background (an approximation of the calibrated cases)."""
    size_in = size_mm / MM_PER_IN
    if len(faults) <= 1:
        f = faults[0] if faults else "normal"
        x, _ = simulate(bearing, f, 0.0 if f == "normal" else size_in, rpm,
                        with_size(cfg, f, size_in) if f != "normal" else cfg, load_hp, seed)
        return x, ({faults[0]: x} if faults else {})
    bg_scale = float(np.mean([cfg.case_bg.get(f"{f}@{size_in:.3f}",
                                              with_size(cfg, f, size_in).case_bg.get(f"{f}@{size_in:.3f}", 1.0))
                              for f in faults])) if cfg.case_bg else 1.0
    bg_cfg = replace(cfg, bg_lines=(), case_eq={}, noise_std=cfg.noise_std * bg_scale,
                     coloured_noise_std=cfg.coloured_noise_std * bg_scale)
    x, _ = simulate(bearing, "normal", 0.0, rpm, bg_cfg, load_hp, seed)
    quiet = replace(cfg, noise_std=0.0, coloured_noise_std=0.0, shaft_harm_amp=(0, 0, 0))
    comps = {}
    for i, f in enumerate(faults):
        xf, _ = simulate(bearing, f, size_in, rpm, with_size(quiet, f, size_in), load_hp, seed + 1 + i)
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
        out = {}
        for k, v in m.items():
            if k.startswith("__"):
                continue
            v = np.squeeze(np.asarray(v, float))
            if v.ndim == 1 and v.size > 1000:
                out[k] = v
            elif v.ndim == 2 and min(v.shape) <= 8 and max(v.shape) > 1000:   # e.g. SpectraQuest (N, 3)
                v = v if v.shape[0] > v.shape[1] else v.T
                for a in range(v.shape[1]):
                    out[f"{k} [axis {a + 1}]"] = v[:, a]
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
is_sq = preset.startswith("SpectraQuest")
calib_kind = "sq" if is_sq else ("cwru" if preset.startswith("CWRU 6205") else None)
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
if is_sq:
    sev = st.sidebar.selectbox("Severity", ["L", "M", "H"], index=2,
                               help="Low / medium / high, as in the SpectraQuest file names.")
    fault_size_mm = SQ_SEV_IN[sev] * MM_PER_IN     # internal severity index, not a real size
    size_label = f"severity {sev}"
    st.sidebar.caption("Severity is categorical here: the defect sizes are not recorded in the data.")
else:
    fault_size_mm = st.sidebar.number_input("Fault size [mm]", min_value=0.05, max_value=3.0,
                                            value=0.5334, step=0.0254, format="%.4f")
    size_label = f"fault size {fault_size_mm} mm"
    st.sidebar.caption(f"= {fault_size_mm / MM_PER_IN:.4f} in   (CWRU: 0.1778 / 0.3556 / 0.5334 / 0.7112 mm)")

# ----------------------------------------------------- realism / calibration
st.sidebar.header("Signal realism")
calib_set = load_calibrated(calib_kind) if calib_kind else {}
calib = None
use_calib = False
if calib_set:
    dname = "SpectraQuest" if is_sq else "CWRU"
    use_calib = st.sidebar.checkbox(f"Use {dname}-calibrated parameters", False)
    if use_calib:
        label = next(iter(calib_set))
        if len(calib_set) > 1:
            label = st.sidebar.selectbox("Accelerometer axis", list(calib_set))
        calib = calib_set[label]
        st.sidebar.caption("Resonances, noise, slip and per-case amplitudes come from the real-data fit "
                           "(sensor units). Manual noise/resonance settings below are ignored. "
                           + ("Each axis is calibrated independently." if is_sq else
                              "Amplitudes are interpolated between 0.18/0.36/0.53 mm."))
elif calib_kind is None and preset != "Custom":
    st.sidebar.caption("No calibration available for this preset.")
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

# ------------------------------------------------------------- realism options
st.sidebar.header("Realism (non-stationarity)")
realism_on = st.sidebar.checkbox(
    "Enable realism effects", True,
    help="Switch everything below off to get the original stationary, perfectly impulsive model.")
override = False
if realism_on and use_calib:
    override = st.sidebar.checkbox(
        "Override fitted per-case values with the sliders", False,
        help="Off: use the values fitted to the real recordings (differ per fault and severity).")
manual = realism_on and (override or not use_calib)
with st.sidebar.expander("Realism controls", expanded=manual):
    if realism_on and not manual:
        st.caption("Using values fitted to the real recordings. Tick the override box above to edit.")
    st.markdown("**Fault signal**")
    r_gfrac = st.slider("Non-impulsive share of fault power", 0.0, 0.99, 0.0, 0.05, disabled=not manual,
                        help="Part of the fault vibration is random (structure-filtered noise) modulated "
                             "at the fault rate instead of clean impacts. Lowers kurtosis and "
                             "fault peaks at equal RMS. Real CWRU and SpectraQuest files need this.")
    r_kappa = st.slider("Sharpness of that modulation", 0.0, 8.0, 3.0, 0.5, disabled=not manual)
    r_drift = st.slider("Slow amplitude drift (log-std)", 0.0, 1.5, 0.0, 0.1, disabled=not manual,
                        help="Fault strength wanders over time; produces the file-to-file spread of kurtosis.")
    r_drift_tau = st.slider("Drift correlation time [s]", 0.05, 5.0, 0.5, 0.05, disabled=not manual)
    st.markdown("**Background (also in healthy bearings)**")
    r_burst_rate = st.slider("Random transients [per s]", 0.0, 40.0, 0.0, 1.0, disabled=not manual)
    r_burst_amp = st.slider("Transient size (x noise RMS)", 0.0, 12.0, 4.0, 0.5, disabled=not manual)
    r_tail = st.select_slider("Heavy-tailed noise (Student-t dof, 0 = Gaussian)",
                              options=[0, 3, 4, 5, 6, 8, 12, 30], value=0, disabled=not manual)
    r_nmod = st.slider("Slow noise-level modulation", 0.0, 1.0, 0.0, 0.1, disabled=not manual)
    r_nmod_tau = st.slider("Noise modulation time [s]", 0.1, 5.0, 1.0, 0.1, disabled=not manual)

st.sidebar.header("Recorder and background")
use_eq = use_shape = use_lines = True
aa_default = float(calib.aa_fc) if (use_calib and calib is not None) else 0.0
if use_calib:
    use_eq = st.sidebar.checkbox("Spectral correction fitted to real spectra", True,
                                 help="Smooth, power-preserving transfer-function correction per fault case "
                                      "(fixes the spectral balance the resonance modes cannot capture).")
    use_shape = st.sidebar.checkbox("Background spectrum shaped like the real recording", True,
                                    help="Off: plain white background noise of the same power.")
    use_lines = st.sidebar.checkbox("Interference lines of the real healthy recording", True,
                                    help="Narrow lines (e.g. ~1.04 kHz in the CWRU normal file); healthy class only.")
aa_cut = st.sidebar.slider("Recorder anti-aliasing low-pass edge [Hz] (0 = none)", 0, int(fs * 0.49),
                           int(min(aa_default, fs * 0.49)), 50, key=f"aa_{preset}_{use_calib}",
                           help="Real CWRU data falls ~60 dB between 5.4 and 5.9 kHz; the calibration fits this edge.")

st.sidebar.header("Visualization")
zoom_ms = st.sidebar.slider("Time-plot window [ms]", 20, 1000, 200)
lo_def = int(min(P["band"][0], 0.40 * fs))
hi_def = int(min(P["band"][1], 0.47 * fs))
band = st.sidebar.slider("Envelope demodulation band [Hz]", 200, int(fs * 0.48), (lo_def, hi_def),
                         key=f"band_{preset}_{fs}")
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
    if fs != calib.fs:
        st.warning(f"This calibration was fitted at fs = {calib.fs} Hz; other rates change the "
                   "resonance response.")
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
    if is_sq:   # severity index 0.001-0.003: reference = L, no entry/exit double impulses
        cfg = replace(cfg, size_ref_in=0.001, double_impulse_size_in=1e9)

# ------------------------------------------------------ apply realism options
_zero = dict(gfrac=0.0, fault_gfrac={}, case_gfrac={}, amp_drift_std=0.0, fault_drift={}, case_drift={},
             bg_burst_rate=0.0, bg_burst_amp=0.0, noise_tail_dof=0.0, noise_mod_depth=0.0,
             case_kappa={}, case_drift_tau={}, case_mod={}, case_bg={})
if not realism_on:
    cfg = replace(cfg, **_zero, aa_fc=0.0, bg_shape_f=(), bg_shape_db=(), bg_lines=(), case_eq={})
else:
    if manual:
        noise_rms = float(np.sqrt(cfg.noise_std ** 2 + cfg.coloured_noise_std ** 2))
        cfg = replace(cfg, **{**_zero, "gfrac": r_gfrac, "amp_drift_std": r_drift,
                              "bg_burst_rate": r_burst_rate, "bg_burst_amp": r_burst_amp * noise_rms,
                              "noise_tail_dof": float(r_tail), "noise_mod_depth": r_nmod},
                      gauss_kappa=r_kappa, amp_drift_tau=r_drift_tau, noise_mod_tau=r_nmod_tau)
    if use_calib:
        if not use_eq:
            cfg = replace(cfg, case_eq={})
        if not use_shape:
            cfg = replace(cfg, bg_shape_f=(), bg_shape_db=())
        if not use_lines:
            cfg = replace(cfg, bg_lines=())
    cfg = replace(cfg, aa_fc=float(aa_cut))

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
    ax.set(xlabel="Time [s]", ylabel="Amplitude", title=f"Simulated signal - {size_label}")
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
    b_sizes = c2.text_input("Severities (L, M, H)" if is_sq else "Fault sizes [mm]",
                            "L, M, H" if is_sq else "0.1778, 0.3556, 0.5334")
    b_segs = int(c3.number_input("Windows per class/size", min_value=1, value=50))
    b_recs = int(c3.number_input("Independent records per class/size", min_value=1, value=5,
                                 help="Each record is a separate simulation (own random seed). Split train/test by "
                                      "record_id, not by window, to avoid leakage between neighbouring windows."))
    b_win = int(c3.number_input("Window length [samples]", min_value=256, value=2048, step=256))
    if st.button("Generate dataset"):
        try:
            if is_sq:   # (label, internal size in mm)
                sizes = [(s.strip().upper(), SQ_SEV_IN[s.strip().upper()] * MM_PER_IN)
                         for s in b_sizes.split(",") if s.strip()]
            else:
                sizes = [(float(s), float(s)) for s in b_sizes.split(",") if s.strip()]
        except (ValueError, KeyError):
            sizes = []
            st.error("Could not parse the fault sizes.")
        combos = [(f, s) for f in b_faults for s in ([("-" if is_sq else 0.0, 0.0)] if f == "normal" else sizes)]
        if combos:
            Xs, lab, sz, rid = [], [], [], []
            bar = st.progress(0.0)
            n_rec = max(1, min(b_recs, b_segs))
            per_rec = [b_segs // n_rec + (1 if j < b_segs % n_rec else 0) for j in range(n_rec)]
            total = len(combos) * n_rec
            done = 0
            for i, (f, s) in enumerate(combos):
                for j, nw in enumerate(per_rec):
                    cfg_b = replace(cfg, duration=nw * b_win / fs)
                    xb, _ = generate(bearing, [] if f == "normal" else [f], s[1], rpm, cfg_b,
                                     load_hp, seed + 1000 * (i + 1) + 17 * j)
                    Xs.append(xb[: nw * b_win].reshape(nw, b_win))
                    lab += [f] * nw
                    sz += [s[0]] * nw
                    rid += [i * n_rec + j] * nw
                    done += 1
                    bar.progress(done / total)
            buf = io.BytesIO()
            size_key = "severity" if is_sq else "size_mm"
            np.savez_compressed(buf, X=np.concatenate(Xs).astype(np.float32), fault=np.array(lab),
                                rpm=rpm, fs=fs, record_id=np.array(rid), **{size_key: np.array(sz)})
            st.session_state["batch_npz"] = buf.getvalue()
            st.session_state["batch_info"] = f"{len(lab)} windows x {b_win} samples, {len(combos)} class/size combos x {n_rec} records"
    if "batch_npz" in st.session_state:
        st.success(st.session_state["batch_info"])
        st.download_button("Download dataset (.npz: X, fault, size_mm or severity, record_id, rpm, fs)",
                           st.session_state["batch_npz"], "bearing_synthetic_dataset.npz")

# ----------------------------------------------------------- downloads
df = pd.DataFrame({"Time": t, "Signal": x})
st.download_button("Download signal CSV", df.to_csv(index=False, float_format="%.6g").encode(),
                   "vibration_signal.csv", "text/csv")
