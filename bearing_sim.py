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
        # random impact sign (not all impacts add in-phase at the sensor)
        # place impulse responses with fractional-sample timing
        L = len(h)
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
                x[i0:i0 + L] += Ai * hh / (len(cfg.modes) and 1.0)
    x = x[:N]

    # deterministic shaft-related components
    for k, amp in enumerate(cfg.shaft_harm_amp, start=1):
        x += amp * np.sin(2 * np.pi * k * fr * t + rng.uniform(0, 2 * np.pi))

    # background noise: white + coloured
    x += rng.normal(0, cfg.noise_std, N)
    bl, al = signal.butter(2, 1500 / (fs / 2))
    cn = signal.lfilter(bl, al, rng.normal(0, 1, N))
    x += cfg.coloured_noise_std * cn / (cn.std() + 1e-12)

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
