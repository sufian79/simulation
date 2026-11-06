"""
Physics-based synthetic bearing fault vibration simulator.

Model:  x(t) = sum_i A_i * h(t - T_i - tau_i)  +  shaft harmonics  +  noise

- T_i   : ideal impact times from kinematic fault frequency (BPFO/BPFI/2*BSF)
- tau_i : random slip/jitter (cumulative, cyclostationary behaviour)
- A_i   : fault-specific amplitude modulation (load zone / cage rotation)
- h(t)  : multi-mode structural impulse response (decaying sinusoids)
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import signal, stats


# ----------------------------------------------------------------- geometry
@dataclass
class Bearing:
    name: str
    n_balls: int
    d: float          # ball diameter (mm)
    D: float          # pitch diameter (mm)
    contact_deg: float = 0.0

    def freqs(self, fr):
        """Fault frequencies (Hz) for shaft frequency fr (Hz)."""
        r = (self.d / self.D) * np.cos(np.radians(self.contact_deg))
        n = self.n_balls
        return dict(
            BPFO=n * fr / 2 * (1 - r),
            BPFI=n * fr / 2 * (1 + r),
            BSF=self.D * fr / (2 * self.d) * (1 - r ** 2),
            FTF=fr / 2 * (1 - r),
        )


CWRU_DE = Bearing("SKF6205-DE", 9, 7.94, 39.04)   # drive end
CWRU_FE = Bearing("SKF6203-FE", 9, 6.75, 28.50)   # fan end
# SpectraQuest MFS, Rexnord ER-12K. 8 balls, d/D = 0.25 (5/16 in balls, 1.25 in pitch).
# Verified against the data: envelope harmonics give BPFO = 3.00x, BPFI = 5.00x shaft speed.
ER12K = Bearing("Rexnord-ER12K", 8, 7.94, 31.75)


# ------------------------------------------------------------------- config
@dataclass
class SimConfig:
    fs: int = 12000
    duration: float = 10.0
    # structural resonance modes: (freq Hz, damping ratio, relative gain)
    modes: tuple = ((3200, 0.06, 1.0), (4500, 0.05, 0.5), (1800, 0.08, 0.4))
    # amplitude vs fault size: A = a0 * (size/size_ref)**p   (calibrate!)
    a0: float = 1.0
    size_ref_in: float = 0.007
    size_exponent: float = 0.8
    jitter: float = 0.01          # std of slip as fraction of impact period
    speed_fluct: float = 0.005    # relative slow speed variation
    ir_amp_scatter: float = 0.15  # random per-impact amplitude scatter (rel.)
    ir_len_ms: float = 8.0
    shaft_harm_amp: tuple = (0.05, 0.02, 0.01)   # 1x,2x,3x
    noise_std: float = 0.05
    coloured_noise_std: float = 0.03   # low-pass (structure/background) noise
    inner_mod_depth: float = 0.8       # load-zone modulation at 1x
    ball_mod_depth: float = 0.8        # cage modulation at FTF
    double_impulse_size_in: float = 0.014  # entry/exit events above this size
    healthy_impacts: bool = False
    fault_gain: dict = field(default_factory=lambda: {'outer': 1.0, 'inner': 1.0, 'ball': 1.0})
    # per-case overrides (real CWRU amplitude is NOT a clean power law of size)
    case_gain: dict = field(default_factory=dict)      # "outer@0.007" -> amplitude
    fault_modes: dict = field(default_factory=dict)    # fault -> modes tuple
    fault_jitter: dict = field(default_factory=dict)   # fault -> jitter
    fault_scatter: dict = field(default_factory=dict)  # fault -> amp scatter
    env_band: dict = field(default_factory=dict)       # fault -> [lo, hi] demod band (Hz)
    # ---- realism / non-stationarity (defaults = off, i.e. the original stationary model)
    amp_drift_std: float = 0.0     # log-std of slow drift of impact amplitude (power preserving)
    amp_drift_tau: float = 0.5     # correlation time of that drift [s]
    fault_drift: dict = field(default_factory=dict)    # fault -> drift std
    case_drift: dict = field(default_factory=dict)     # "outer@0.007" -> drift std
    bg_burst_rate: float = 0.0     # random background transients per second (any bearing state)
    bg_burst_amp: float = 0.0      # typical transient amplitude (signal units)
    bg_burst_sigma: float = 0.5    # log-normal spread of transient amplitudes
    noise_tail_dof: float = 0.0    # 0 = Gaussian noise, else Student-t degrees of freedom (>2)
    noise_mod_depth: float = 0.0   # log-std of slow noise-level modulation (power preserving)
    noise_mod_tau: float = 1.0     # correlation time of that modulation [s]
    # share of the fault's vibration power carried by NON-impulsive, fault-modulated random
    # vibration (structure-filtered Gaussian noise, amplitude-modulated at the fault rate)
    gfrac: float = 0.0
    fault_gfrac: dict = field(default_factory=dict)
    case_gfrac: dict = field(default_factory=dict)
    gauss_kappa: float = 3.0       # sharpness of that modulation (0 = unmodulated)


# ---------------------------------------------------------------- simulator
def impulse_response(cfg: SimConfig):
    n = int(cfg.ir_len_ms * 1e-3 * cfg.fs)
    t = np.arange(n) / cfg.fs
    h = np.zeros(n)
    for f, z, g in cfg.modes:
        wn = 2 * np.pi * f
        wd = wn * np.sqrt(1 - z ** 2)
        h += g * np.exp(-z * wn * t) * np.sin(wd * t)
    return h / np.max(np.abs(h))


def _ar1(N, fs, tau, rng):
    """Unit-variance, slowly varying random process with correlation time tau [s]."""
    a = float(np.exp(-1.0 / (max(tau, 1e-3) * fs)))
    return signal.lfilter([np.sqrt(1 - a * a)], [1.0, -a], rng.normal(0, 1, N))


def _add_responses(x, T, A, modes, fs, L):
    """Add A_i * h(t - T_i) (sum of decaying resonance modes, fractional-sample timing)."""
    tt = np.arange(L) / fs
    for Ti, Ai in zip(T, A):
        i0 = int(Ti * fs)
        frac = Ti - i0 / fs
        hh = np.zeros(L)
        m = tt >= frac
        ts = tt[m] - frac
        for f, z, g in modes:
            wn = 2 * np.pi * f
            wd = wn * np.sqrt(1 - z ** 2)
            hh[m] += g * np.exp(-z * wn * ts) * np.sin(wd * ts)
        if i0 + L < x.size:
            x[i0:i0 + L] += Ai * hh


def _event_times(rate, fs, N, jitter, rng, speed_profile):
    """Impact times (s) with cumulative speed variation + per-event slip."""
    # phase of defect = integral of rate * speed_profile
    dur = N / fs
    phase = np.cumsum(speed_profile) / fs * rate
    n_ev = int(phase[-1]) + 1
    k = np.arange(1, n_ev + 1)
    T = np.interp(k, phase, np.arange(N) / fs)
    T = T[T < dur - 0.01]
    period = 1.0 / rate
    T = T + rng.normal(0, jitter * period, size=T.size)
    return np.sort(T)


def simulate(bearing: Bearing, fault: str, size_in: float, rpm: float,
             cfg: SimConfig = SimConfig(), load_hp: float = 0.0,
             seed: int | None = None):
    """
    fault: 'normal' | 'outer' | 'inner' | 'ball'
    Returns (x, meta). x is acceleration-like signal (arb. units).
    """
    rng = np.random.default_rng(seed)
    rng_x = np.random.default_rng(None if seed is None else [int(seed), 777])
    fs = cfg.fs
    N = int(cfg.duration * fs)
    t = np.arange(N) / fs
    fr = rpm / 60.0

    # slowly varying speed (low-pass filtered noise)
    sp = rng.normal(0, 1, N)
    b, a = signal.butter(2, 1.0 / (fs / 2))
    sp = signal.filtfilt(b, a, sp)
    sp = 1 + cfg.speed_fluct * sp / (sp.std() + 1e-12)

    ff = bearing.freqs(fr)
    h = impulse_response(cfg)
    x = np.zeros(N + len(h))

    # amplitude: per-case calibrated value if available, else size power law
    ck = f"{fault}@{size_in:.3f}"
    if fault == "normal":
        base = 0.0
    elif ck in cfg.case_gain:
        base = cfg.case_gain[ck]
    else:
        base = (cfg.a0 * (size_in / cfg.size_ref_in) ** cfg.size_exponent
                * cfg.fault_gain.get(fault, 1.0))
    base *= (1 + 0.04 * load_hp)
    modes = cfg.fault_modes.get(fault, cfg.modes)
    jit = cfg.fault_jitter.get(fault, cfg.jitter)
    scat = cfg.fault_scatter.get(fault, cfg.ir_amp_scatter)

    if fault != "normal":
        if fault == "outer":
            rate = ff["BPFO"]
        elif fault == "inner":
            rate = ff["BPFI"]
        elif fault == "ball":
            rate = 2 * ff["BSF"]         # ball strikes both races per rotation
        else:
            raise ValueError(fault)

        T = _event_times(rate, fs, N, jit, rng, sp)
        # double-impulse (entry + exit) for large defects
        if size_in >= cfg.double_impulse_size_in:
            # defect arc length / rolling speed -> delay between the two events
            delay = (size_in * 25.4) / (np.pi * bearing.D * fr * 0.4) * 0.25
            T = np.sort(np.concatenate([T, T + delay]))
            amp_scale = np.concatenate([np.ones(0)])  # placeholder
        # amplitude modulation
        if fault == "inner":
            mod = 1 + cfg.inner_mod_depth * np.cos(2 * np.pi * fr * T)
        elif fault == "ball":
            mod = 1 + cfg.ball_mod_depth * np.cos(2 * np.pi * ff["FTF"] * T)
        else:
            mod = np.ones_like(T)
        A = base * np.clip(mod, 0, None)
        A *= 1 + scat * rng.normal(0, 1, T.size)
        # slow amplitude drift (non-stationarity); exp(d z - d^2) keeps mean power unchanged
        dstd = cfg.case_drift.get(ck, cfg.fault_drift.get(fault, cfg.amp_drift_std))
        if dstd > 0:
            z = _ar1(N, fs, cfg.amp_drift_tau, rng_x)
            idx = np.clip((T * fs).astype(int), 0, N - 1)
            w = np.exp(dstd * z - dstd ** 2)
            A = A * w[idx] / np.sqrt(np.mean(w ** 2))   # realised power fixed, shape drifts
        _add_responses(x, T, A, modes, fs, len(h))
        r = cfg.case_gfrac.get(ck, cfg.fault_gfrac.get(fault, cfg.gfrac))
        if r > 0:
            r = min(float(r), 0.995)
            Pi = x[:N].var()                      # total fault power is preserved by the mix
            tl = np.arange(len(h)) / fs
            hm = np.zeros(len(h))
            for f_, z_, g_ in modes:
                w_ = 2 * np.pi * f_
                hm += g_ * np.exp(-z_ * w_ * tl) * np.sin(w_ * np.sqrt(1 - z_ ** 2) * tl)
            xg = signal.fftconvolve(rng_x.normal(0, 1, N), hm)[:N]
            step = 2 * np.pi * jit / np.sqrt(max(fs / rate, 1.0))
            ph = 2 * np.pi * np.cumsum(rate * sp) / fs + np.cumsum(rng_x.normal(0, step, N))
            env = np.exp(cfg.gauss_kappa * (np.cos(ph) - 1.0))
            if fault == "inner":
                env = env * np.clip(1 + cfg.inner_mod_depth * np.cos(2 * np.pi * fr * t), 0, None)
            elif fault == "ball":
                env = env * np.clip(1 + cfg.ball_mod_depth * np.cos(2 * np.pi * ff["FTF"] * t), 0, None)
            xg = xg * env
            xg *= np.sqrt(Pi / (xg.var() + 1e-30))
            x[:N] = np.sqrt(1 - r) * x[:N] + np.sqrt(r) * xg

    # random background transients (present in healthy bearings too)
    if cfg.bg_burst_rate > 0 and cfg.bg_burst_amp > 0:
        nb = int(rng_x.poisson(cfg.bg_burst_rate * N / fs))
        if nb:
            Tb = rng_x.uniform(0, N / fs - 0.01, nb)
            Ab = (cfg.bg_burst_amp * rng_x.lognormal(0, cfg.bg_burst_sigma, nb)
                  * rng_x.choice([-1.0, 1.0], nb))
            _add_responses(x, Tb, Ab, cfg.modes, fs, len(h))
    x = x[:N]

    # deterministic shaft-related components
    for k, amp in enumerate(cfg.shaft_harm_amp, start=1):
        x += amp * np.sin(2 * np.pi * k * fr * t + rng.uniform(0, 2 * np.pi))

    # background noise: white (Gaussian or heavy-tailed) + coloured
    if cfg.noise_tail_dof and cfg.noise_tail_dof > 2.0:
        dof = float(cfg.noise_tail_dof)
        w = rng_x.standard_t(dof, N) / np.sqrt(dof / (dof - 2.0))
    else:
        w = rng.normal(0, 1, N)
    bl, al = signal.butter(2, 1500 / (fs / 2))
    cn = signal.lfilter(bl, al, rng.normal(0, 1, N))
    noise = cfg.noise_std * w + cfg.coloured_noise_std * cn / (cn.std() + 1e-12)
    if cfg.noise_mod_depth > 0:   # slow non-stationary noise level
        zn = _ar1(N, fs, cfg.noise_mod_tau, rng_x)
        noise = noise * np.exp(cfg.noise_mod_depth * zn - cfg.noise_mod_depth ** 2)
    x += noise

    meta = dict(fault=fault, size_in=size_in, rpm=rpm, fs=fs,
                load_hp=load_hp, freqs=ff, bearing=bearing.name)
    return x, meta


# --------------------------------------------------------------- validation
def envelope_spectrum(x, fs, band=(2000, 5000)):
    sos = signal.butter(4, band, btype="band", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, x)
    env = np.abs(signal.hilbert(xb))
    env -= env.mean()
    win = np.hanning(len(env))
    spec = np.abs(np.fft.rfft(env * win)) / len(env)
    f = np.fft.rfftfreq(len(env), 1 / fs)
    return f, spec


def time_features(x):
    rms = np.sqrt(np.mean(x ** 2))
    return dict(rms=rms, kurtosis=stats.kurtosis(x, fisher=False),
                crest=np.max(np.abs(x)) / rms, skew=stats.skew(x),
                p2p=np.ptp(x))


def check_peaks(x, fs, meta, tol=0.02, nharm=3):
    """Does the envelope spectrum show peaks at the expected fault freqs?"""
    f, s = envelope_spectrum(x, fs)
    ff = meta["freqs"]
    key = {"outer": "BPFO", "inner": "BPFI", "ball": "BSF"}.get(meta["fault"])
    if key is None:
        return {}
    f0 = ff[key] * (2 if key == "BSF" else 1)
    floor = np.median(s[(f > 5) & (f < 600)])
    out = {}
    for k in range(1, nharm + 1):
        fk = k * f0
        m = (f > fk * (1 - tol)) & (f < fk * (1 + tol))
        out[f"{k}x{key}"] = (round(fk, 1), round(float(s[m].max() / floor), 1))
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = SimConfig(duration=10)
    cases = [("normal", 0.0), ("outer", 0.007), ("inner", 0.014),
             ("ball", 0.021)]
    fig, ax = plt.subplots(len(cases), 2, figsize=(13, 9))
    for r, (flt, sz) in enumerate(cases):
        x, meta = simulate(CWRU_DE, flt, sz, rpm=1797, cfg=cfg, seed=r)
        tf = time_features(x)
        print(flt, sz, {k: round(v, 3) for k, v in tf.items()})
        print("   peaks (freq, SNR-vs-floor):", check_peaks(x, cfg.fs, meta))
        ax[r, 0].plot(np.arange(3000) / cfg.fs, x[:3000], lw=0.6)
        ax[r, 0].set_title(f"{flt} {sz} in  |  kurt={tf['kurtosis']:.2f}")
        f, s = envelope_spectrum(x, cfg.fs)
        ax[r, 1].plot(f, s, lw=0.6)
        ax[r, 1].set_xlim(0, 600)
        ax[r, 1].set_title("envelope spectrum")
        for k, v in meta["freqs"].items():
            if k in ("BPFO", "BPFI"):
                ax[r, 1].axvline(v, color="r", ls=":", lw=0.7)
    plt.tight_layout()
    plt.savefig("sim_check.png", dpi=110)
    print("Fault freqs @1797rpm:", {k: round(v, 2) for k, v in
          CWRU_DE.freqs(1797 / 60).items()})
