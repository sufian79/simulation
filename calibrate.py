"""
Calibrate SimConfig against real bearing data (CWRU-style).

Stage A (analytic, fast):
  1. noise level + shaft harmonics      <- normal file
  2. resonance modes (freq, damping)    <- excess PSD of fault files vs normal
  3. amplitude-vs-size exponent + per-fault gain  <- excess power, log-log fit
Stage B (refinement, small grid search):
  4. jitter and impact-amplitude scatter <- kurtosis, crest, envelope-peak ratios

Usage (real data):
    python calibrate.py --data_dir /path/to/cwru_mats --out calibrated.json
Self-test (parameter recovery on synthetic "real" data):
    python calibrate.py --selftest
"""
import argparse, json, re, itertools, os
from dataclasses import replace, asdict
import numpy as np
from scipy import signal, io as sio, stats
from scipy.optimize import least_squares
from bearing_sim import (Bearing, CWRU_DE, ER12K, SimConfig, simulate,
                         envelope_spectrum, time_features)

# CWRU 12 kHz drive-end, 0 hp : file number -> (fault, size_in)
CWRU_FILES = {97: ("normal", 0.0),
              105: ("inner", .007), 169: ("inner", .014), 209: ("inner", .021),
              118: ("ball", .007), 185: ("ball", .014), 222: ("ball", .021),
              130: ("outer", .007), 197: ("outer", .014), 234: ("outer", .021)}


# ------------------------------------------------------------------ loading
def load_cwru_dir(data_dir, default_rpm=1797.0, channel="DE_time"):
    """Return list of dict(x, fault, size, rpm, fs) from .mat files whose
    name contains a known CWRU file number."""
    out = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".mat"):
            continue
        nums = [int(n) for n in re.findall(r"\d+", fn)]
        key = next((n for n in nums if n in CWRU_FILES), None)
        if key is None:
            continue
        m = sio.loadmat(os.path.join(data_dir, fn))
        k = next((k for k in m if k.endswith(channel)), None)
        if k is None:
            continue
        rk = next((k2 for k2 in m if k2.endswith("RPM")), None)
        rpm = float(np.squeeze(m[rk])) if rk else default_rpm
        flt, sz = CWRU_FILES[key]
        out.append(dict(x=np.squeeze(m[k]).astype(float), fault=flt,
                        size=sz, rpm=rpm, fs=12000, file=key))
    return out


SQ_SEV = {"L": 0.001, "M": 0.002, "H": 0.003}   # severity index/1000 (no physical size known)
SQ_FAULT = {"BD": "ball", "ID": "inner", "OD": "outer"}


def load_spectraquest_dir(data_dir, axis=0, seconds=15.0, fs=25600, rpm=1200.0):
    """SpectraQuest files '20Hz_<Normal|BD|ID|OD><L|M|H>.mat': (N, 3) array, one
    column per axis. Severity L/M/H is stored as size 0.001/0.002/0.003."""
    out = []
    for fn in sorted(os.listdir(data_dir)):
        mt = re.match(r"20Hz_(Normal|BD[LMH]|ID[LMH]|OD[LMH])\.mat$", fn, re.I)
        if not mt:
            continue
        tag = mt.group(1)
        m = sio.loadmat(os.path.join(data_dir, fn))
        k = next(k for k in m if not k.startswith("__"))
        x = np.asarray(m[k])[: int(seconds * fs), axis].astype(float)
        if tag.lower() == "normal":
            flt, sz = "normal", 0.0
        else:
            flt, sz = SQ_FAULT[tag[:2].upper()], SQ_SEV[tag[2].upper()]
        out.append(dict(x=x, fault=flt, size=sz, rpm=rpm, fs=fs, file=tag))
    return out


# --------------------------------------------------------------- stage A
def _welch(x, fs):
    return signal.welch(x, fs, nperseg=2048, noverlap=1024)


def fit_noise_and_harmonics(rec, nharm=3):
    x, fs, rpm = rec["x"] - rec["x"].mean(), rec["fs"], rec["rpm"]
    fr = rpm / 60
    N = len(x)
    win = np.hanning(N)
    X = np.abs(np.fft.rfft(x * win)) * 2 / win.sum()
    f = np.fft.rfftfreq(N, 1 / fs)
    amps = []
    for k in range(1, nharm + 1):
        m = (f > k * fr * 0.98) & (f < k * fr * 1.02)
        amps.append(float(X[m].max()))
    p_total = np.mean(x ** 2)
    p_noise = max(p_total - 0.5 * sum(a * a for a in amps), 1e-12)
    ff, P = _welch(x, fs)
    frac_low = P[ff < 1500].sum() / P.sum()
    # white noise puts 1500/(fs/2) of its power below 1500 Hz, coloured ~90%
    w_low = min(1500 / (fs / 2), 0.95)
    c_share = float(np.clip((frac_low - w_low) / (0.9 - w_low), 0.0, 1.0))
    return dict(noise_std=float(np.sqrt(p_noise * (1 - c_share))),
                coloured_noise_std=float(np.sqrt(p_noise * c_share)),
                shaft_harm_amp=tuple(round(a, 4) for a in amps))


def fit_modes(normal, faults, nmodes=3, fmin=400, zmax=0.12):
    """Resonance modes from the excess PSD (fault minus normal), pooled over
    the given fault files, each weighted by sqrt(excess power)."""
    fs = normal["fs"]
    ff, Pn = _welch(normal["x"], fs)
    acc = np.zeros_like(Pn)
    for r in faults:
        _, P = _welch(r["x"], fs)
        d = np.clip(P - Pn, 0, None)
        acc += d / (d.sum() + 1e-18) * np.sqrt(d.sum())
    acc = signal.savgol_filter(acc, 9, 2)
    df = ff[1] - ff[0]
    band = (ff > fmin) & (ff < 0.95 * fs / 2)
    pk, _ = signal.find_peaks(np.where(band, acc, 0),
                              prominence=0.05 * acc[band].max(),
                              distance=int(800 / df))
    order = np.argsort(acc[pk])[::-1][:nmodes]
    pk = pk[order]
    w = signal.peak_widths(acc, pk, rel_height=0.5)[0] * df
    raw = []
    for i, p in enumerate(pk):
        z = float(np.clip(w[i] / (2 * ff[p]), 0.02, zmax))
        # PSD peak of exp(-z*w*t)*sin(wd*t) scales as g^2 / (z*f)^2
        raw.append((float(ff[p]), z, np.sqrt(acc[p]) * z * ff[p]))
    gmax = max(r[2] for r in raw)
    modes = [(round(f, 0), round(z, 3), round(float(g / gmax), 3)) for f, z, g in raw]
    modes.sort(key=lambda m: -m[2])
    return tuple(modes)


def _excess_psd(normal, faults):
    """Pooled excess PSD (fault minus normal), each file weighted by sqrt(excess power)."""
    fs = normal["fs"]
    ff, Pn = _welch(normal["x"], fs)
    acc = np.zeros_like(Pn)
    for r in faults:
        _, P = _welch(r["x"], fs)
        d = np.clip(P - Pn, 0, None)
        acc += d / (d.sum() + 1e-18) * np.sqrt(d.sum())
    return ff, np.clip(signal.savgol_filter(acc, 9, 2), 0.0, None)


def _mode_ir(p, K, fs, L):
    t = np.arange(L) / fs
    h = np.zeros(L)
    for i in range(K):
        f, z, g = np.exp(p[3 * i:3 * i + 3])
        wn = 2 * np.pi * f
        h += g * np.exp(-z * wn * t) * np.sin(wn * np.sqrt(1 - z * z) * t)
    return h


def fit_modes_psd(normal, faults, init, K=4, fmin=400, ir_len_ms=8.0, nfft=2048):
    """Refine resonance modes by least squares on the LOG excess PSD, using the same
    finite impulse response the simulator uses (so the fitted spectrum is what it will produce).
    Returns (modes, (err_db_before, err_db_after))."""
    fs = normal["fs"]
    L = int(ir_len_ms * 1e-3 * fs)
    ff, acc = _excess_psd(normal, faults)
    sel = (ff > fmin) & (ff < 0.95 * fs / 2)
    eps = 1e-3 * acc[sel].max()
    target = np.log(acc[sel] + eps)
    modes0 = list(init)[:K]
    spare = np.linspace(0.15, 0.45, K) * fs
    while len(modes0) < K:
        modes0.append((float(spare[len(modes0)]), 0.08, 0.1))
    p0 = []
    for f, z, g in modes0:
        p0 += [np.log(f), np.log(np.clip(z, 0.015, 0.35)), np.log(max(g, 0.05))]
    p0 = np.array(p0 + [0.0])

    def model(p):
        return np.exp(p[-1]) * np.abs(np.fft.rfft(_mode_ir(p, K, fs, L), n=nfft)) ** 2

    p0[-1] = np.log(acc[sel].max() / model(p0)[sel].max())
    resid = lambda p: np.log(model(p)[sel] + eps) - target
    lo = np.array([np.log(fmin), np.log(0.012), np.log(0.01)] * K + [-80.0])
    hi = np.array([np.log(0.47 * fs), np.log(0.4), np.log(30.0)] * K + [80.0])
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
    res = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.7, max_nfev=400)
    db = lambda r: float(np.sqrt(np.mean(r ** 2)) * 4.343)
    e0, e1 = db(resid(p0)), db(resid(res.x))
    if e1 >= e0:
        return tuple(init), (e0, e0)
    raw = [(float(np.exp(res.x[3 * i])), float(np.exp(res.x[3 * i + 1])), float(np.exp(res.x[3 * i + 2])))
           for i in range(K)]
    gmax = max(g for _, _, g in raw)
    modes = [(round(f, 0), round(z, 3), round(g / gmax, 3)) for f, z, g in raw if g / gmax > 0.02]
    modes.sort(key=lambda m: -m[2])
    return tuple(modes), (e0, e1)


def fit_background(normal, cfg, bearing, dur_max=10.0):
    """Healthy-bearing non-stationarity: random transients, heavy-tailed noise and slow noise
    modulation, chosen to match the real normal file's kurtosis and crest factor."""
    fs = normal["fs"]
    dur = min(len(normal["x"]) / fs, dur_max)
    x = normal["x"][: int(dur * fs)]
    x = x - x.mean()
    kr, cr, var_r = stats.kurtosis(x, fisher=False), np.abs(x).max() / x.std(), x.var()
    noise_rms = np.sqrt(cfg.noise_std ** 2 + cfg.coloured_noise_std ** 2)
    best, best_cost = None, np.inf
    for dof, rate, am, md in itertools.product((0, 6), (0, 1, 4, 12, 30), (2, 4, 8), (0.0, 0.3, 0.6)):
        if rate == 0 and am != 2:
            continue                      # amplitude is irrelevant without transients
        c = replace(cfg, duration=dur, noise_tail_dof=dof, bg_burst_rate=rate,
                    bg_burst_amp=am * noise_rms, noise_mod_depth=md)
        cost = 0.0
        for sd in (11, 12):
            xs, _ = simulate(bearing, "normal", 0.0, normal["rpm"], c, seed=sd)
            cost += (np.log(stats.kurtosis(xs, fisher=False) / kr) ** 2
                     + np.log(np.abs(xs).max() / xs.std() / cr) ** 2)
        if cost < best_cost:
            best, best_cost = c, cost
    # keep the total healthy power equal to the real file's power
    xq, _ = simulate(bearing, "normal", 0.0, normal["rpm"],
                     replace(best, noise_std=0.0, coloured_noise_std=0.0), seed=21)
    nv = best.noise_std ** 2 + best.coloured_noise_std ** 2
    k = float(np.sqrt(max(var_r - xq.var(), 0.2 * nv) / nv))
    best = replace(best, noise_std=best.noise_std * k, coloured_noise_std=best.coloured_noise_std * k,
                   duration=cfg.duration)
    print(f"background: tail dof={best.noise_tail_dof}, transients {best.bg_burst_rate}/s x "
          f"{best.bg_burst_amp:.3f}, noise modulation {best.noise_mod_depth} "
          f"(healthy kurtosis real {kr:.1f}, crest {cr:.1f})")
    return best


def fit_drift(cfg, faults, bearing, bands, dur=6.0, grid=(0.0, 0.3, 0.6, 1.0, 1.5)):
    """Per-case slow amplitude drift, chosen to match kurtosis / crest / envelope features."""
    out = {}
    for r in faults:
        kind, fs = r["fault"], r["fs"]
        bd = bands.get(kind, (2000, 5000))
        d_ = min(dur, len(r["x"]) / fs)
        _, meta = simulate(bearing, kind, r["size"], r["rpm"], replace(cfg, duration=0.5), seed=0)
        fr = _feat(r["x"][: int(d_ * fs)], fs, meta, bd)
        key = f"{kind}@{r['size']:.3f}"
        best = (np.inf, 0.0)
        for d in grid:
            c = replace(cfg, case_drift={key: d}, duration=d_)
            cost = 0.0
            for sd in (1, 2):
                x, _ = simulate(bearing, kind, r["size"], r["rpm"], c, seed=sd)
                cost += np.sum((_feat(x, fs, meta, bd) - fr) ** 2)
            best = min(best, (cost, d))
        out[key] = float(best[1])
    return out


def fit_gfrac(cfg, faults, bearing, bands, dur=6.0, grid=(0.0, 0.3, 0.6, 0.8, 0.92, 0.97)):
    """Per-case share of fault power that is non-impulsive random vibration, chosen to match
    kurtosis / crest / envelope features of the real file."""
    out = {}
    for r in faults:
        kind, fs = r["fault"], r["fs"]
        bd = bands.get(kind, (2000, 5000))
        d_ = min(dur, len(r["x"]) / fs)
        _, meta = simulate(bearing, kind, r["size"], r["rpm"], replace(cfg, duration=0.5), seed=0)
        fr = _feat(r["x"][: int(d_ * fs)], fs, meta, bd)
        key = f"{kind}@{r['size']:.3f}"
        best = (np.inf, 0.0)
        for g in grid:
            c_ = replace(cfg, case_gfrac={key: g}, duration=d_)
            cost = 0.0
            for sd in (1, 2):
                x, _ = simulate(bearing, kind, r["size"], r["rpm"], c_, seed=sd)
                cost += np.sum((_feat(x, fs, meta, bd) - fr) ** 2)
            best = min(best, (cost, g))
        out[key] = float(best[1])
    return out


def trim_gains(cfg, records, bearing, iters=3, seeds=(31, 32, 33), dur=6.0):
    """Final step: rescale each case's impact amplitude so the simulated RMS (with ALL realism
    components active) equals the real RMS. Corrects power shifts caused by drift / transients."""
    normal = next(r for r in records if r["fault"] == "normal")
    faults = [r for r in records if r["fault"] != "normal"]
    for _ in range(iters):
        gains = dict(cfg.case_gain)
        for r in faults:
            key = f"{r['fault']}@{r['size']:.3f}"
            d_ = min(dur, len(r["x"]) / r["fs"])
            vr = np.var(r["x"][: int(d_ * r["fs"])])
            cc = replace(cfg, duration=d_)
            vs = np.mean([np.var(simulate(bearing, r["fault"], r["size"], r["rpm"], cc, seed=s)[0])
                          for s in seeds])
            vb = np.mean([np.var(simulate(bearing, "normal", 0.0, r["rpm"], cc, seed=s)[0])
                          for s in seeds])
            k = np.sqrt(max(vr - vb, 0.05 * vr) / max(vs - vb, 0.05 * vs))
            gains[key] = float(gains[key] * np.clip(k, 0.3, 3.0))
        cfg = replace(cfg, case_gain=gains)
    return cfg


def fit_amplitudes(cfg, normal, faults, bearing, dur=4.0):
    """Per-case impact amplitude (matches excess power) + power-law diagnostic."""
    p_norm = np.mean((normal["x"] - normal["x"].mean()) ** 2)
    quiet = replace(cfg, noise_std=0.0, coloured_noise_std=0.0,
                    shaft_harm_amp=(0, 0, 0), size_exponent=0.0, a0=1.0,
                    duration=dur, case_gain={},
                    fault_gain={"outer": 1., "inner": 1., "ball": 1.},
                    bg_burst_rate=0.0, noise_tail_dof=0.0, noise_mod_depth=0.0,
                    amp_drift_std=0.0, fault_drift={}, case_drift={},
                    gfrac=0.0, fault_gfrac={}, case_gfrac={})
    case, rows = {}, []
    for r in faults:
        p_ex = max(np.mean((r["x"] - r["x"].mean()) ** 2) - p_norm, 1e-12)
        xu, _ = simulate(bearing, r["fault"], r["size"], r["rpm"], quiet, seed=0)
        a = float(np.sqrt(p_ex / np.mean(xu ** 2)))
        case[f"{r['fault']}@{r['size']:.3f}"] = a
        rows.append((r["fault"], r["size"], np.log(a)))
    kinds = sorted({r[0] for r in rows})
    X = np.array([[1.0 if r[0] == k else 0.0 for k in kinds] +
                  [np.log(r[1] / cfg.size_ref_in)] for r in rows])
    y = np.array([r[2] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    gains = {k: float(np.exp(c)) for k, c in zip(kinds, coef[:-1])}
    return case, float(coef[-1]), gains, float(r2)


# --------------------------------------------------------------- stage B
def _feat(x, fs, meta, band=(2000, 5000)):
    tf = time_features(x)
    f, s = envelope_spectrum(x, fs, band)
    key = {"outer": "BPFO", "inner": "BPFI", "ball": "BSF"}[meta["fault"]]
    f0 = meta["freqs"][key] * (2 if key == "BSF" else 1)
    floor = np.median(s[(f > 5) & (f < 600)])
    pk = lambda fk: s[(f > fk * .98) & (f < fk * 1.02)].max() / floor
    p1, p2 = pk(f0), pk(2 * f0)
    return np.array([np.log(tf["kurtosis"]), np.log(tf["crest"]),
                     np.log(p1), np.log(p2 / p1)])


def refine(cfg, faults, bearing, dur=4.0, bands=None,
           jitters=(0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.5),
           scatters=(0.05, 0.15, 0.3, 0.5)):
    """Per fault type: pick (jitter, amplitude scatter) matching real features."""
    fj, fsc = {}, {}
    bands = bands or {}
    for kind in sorted({r["fault"] for r in faults}):
        bd = bands.get(kind, (2000, 5000))
        sub = [r for r in faults if r["fault"] == kind]
        real = []
        for r in sub:
            _, meta = simulate(bearing, r["fault"], r["size"], r["rpm"],
                               replace(cfg, duration=0.5), seed=0)
            real.append(_feat(r["x"][:int(dur * r["fs"])], r["fs"], meta, bd))
        best, best_cost = None, np.inf
        for j, sc in itertools.product(jitters, scatters):
            c = replace(cfg, fault_jitter={kind: j}, fault_scatter={kind: sc},
                        duration=dur)
            cost = 0.0
            for r, fr in zip(sub, real):
                x, meta = simulate(bearing, r["fault"], r["size"], r["rpm"], c, seed=1)
                cost += np.sum((_feat(x, r["fs"], meta, bd) - fr) ** 2)
            if cost < best_cost:
                best, best_cost = (j, sc), cost
        fj[kind], fsc[kind] = best
        print(f"refine {kind:6s}: jitter={best[0]}, scatter={best[1]}, "
              f"feature cost={best_cost / len(sub):.3f}")
    return fj, fsc


# ----------------------------------------------------------------- driver
def calibrate(records, bearing=CWRU_DE, base=SimConfig(), do_refine=True,
              band=(2000, 5000), fmin=400, zmax=0.12, realism=True):
    """realism=False reproduces the original stationary calibration (v1)."""
    normal = next(r for r in records if r["fault"] == "normal")
    faults = [r for r in records if r["fault"] != "normal"]
    cfg = replace(base, fs=normal["fs"])
    cfg = replace(cfg, **fit_noise_and_harmonics(normal))
    kinds = sorted({r["fault"] for r in faults})
    pooled = fit_modes(normal, faults, fmin=fmin, zmax=zmax)
    per_kind = {k: fit_modes(normal, [r for r in faults if r["fault"] == k], fmin=fmin, zmax=zmax)
                for k in kinds}
    if realism:                      # 1. spectral shape: least squares on the log PSD
        pooled, _ = fit_modes_psd(normal, faults, pooled, fmin=fmin, ir_len_ms=cfg.ir_len_ms)
        for k in kinds:
            per_kind[k], (e0, e1) = fit_modes_psd(normal, [r for r in faults if r["fault"] == k],
                                                  per_kind[k], fmin=fmin, ir_len_ms=cfg.ir_len_ms)
            print(f"spectral fit {k:6s}: log-PSD error {e0:.1f} dB -> {e1:.1f} dB "
                  f"({len(per_kind[k])} modes)")
    cfg = replace(cfg, modes=pooled, fault_modes=per_kind)
    fs_ = normal["fs"]
    if band == "auto":     # demodulate around each fault type's dominant resonance
        bands = {k: (int(max(300, 0.7 * cfg.fault_modes[k][0][0])),
                     int(min(0.48 * fs_, 1.35 * cfg.fault_modes[k][0][0]))) for k in kinds}
    else:
        bands = {k: tuple(band) for k in kinds}
    cfg = replace(cfg, env_band={k: list(v) for k, v in bands.items()})
    if realism:                      # 2. healthy-bearing non-stationarity
        cfg = fit_background(normal, cfg, bearing)
    case, p, gains, r2 = fit_amplitudes(cfg, normal, faults, bearing)
    cfg = replace(cfg, case_gain=case, size_exponent=p, a0=1.0, fault_gain=gains)
    print(f"power-law diagnostic: exponent={p:.2f}, R^2={r2:.2f} "
          f"({'usable' if r2 > 0.8 else 'POOR: use per-case amplitudes'})")
    if do_refine:
        if realism:                  # 3. non-impulsive share of the fault power (per case)
            cg = fit_gfrac(cfg, faults, bearing, bands)
            cfg = replace(cfg, case_gfrac=cg)
            print("non-impulsive share per case:", cg)
        fj, fsc = refine(cfg, faults, bearing, bands=bands)
        cfg = replace(cfg, fault_jitter=fj, fault_scatter=fsc)
        if realism:                  # 4. per-case slow amplitude drift
            cd = fit_drift(cfg, faults, bearing, bands, grid=(0.0, 0.5, 1.0))
            cfg = replace(cfg, case_drift=cd)
            print("drift per case:", cd)
            cfg = trim_gains(cfg, records, bearing)
    return cfg


def _peak_ratio(x, fs, meta, band=(2000, 5000)):
    return float(np.exp(_feat(x, fs, meta, band)[2]))


def report(cfg, records, bearing=CWRU_DE):
    print(f"\n{'case':<14}{'kurt real/sim':>15}{'crest real/sim':>17}"
          f"{'rms real/sim':>15}{'env-peak real/sim':>21}")
    for r in records:
        dur = min(len(r["x"]) / r["fs"], 10.0)
        x, meta = simulate(bearing, r["fault"], r["size"], r["rpm"],
                           replace(cfg, duration=dur), seed=7)
        xr = r["x"][:int(dur * r["fs"])]
        a, b = time_features(xr), time_features(x)
        pk = ("", "")
        if r["fault"] != "normal":
            bd = tuple(cfg.env_band.get(r["fault"], (2000, 5000)))
            pk = (f"{_peak_ratio(xr, r['fs'], meta, bd):.0f}",
                  f"{_peak_ratio(x, r['fs'], meta, bd):.0f}")
        print(f"{r['fault']+' '+str(r['size']):<14}"
              f"{a['kurtosis']:>7.1f}/{b['kurtosis']:<7.1f}"
              f"{a['crest']:>8.1f}/{b['crest']:<8.1f}"
              f"{a['rms']:>7.3f}/{b['rms']:<7.3f}"
              f"{pk[0]:>10}/{pk[1]:<8}")


def selftest():
    truth = SimConfig(modes=((2800, .05, 1.0), (4200, .04, .6), (1500, .07, .3)),
                      size_exponent=1.1, noise_std=0.08, coloured_noise_std=0.05,
                      jitter=0.02, ir_amp_scatter=0.15, duration=10,
                      fault_gain={"outer": 0.8, "inner": 1.6, "ball": 1.2})
    recs = []
    for fl, sz in [("normal", 0)] + [(f, s) for f in ("inner", "ball", "outer")
                                     for s in (.007, .014, .021)]:
        x, _ = simulate(CWRU_DE, fl, sz, 1797, truth, seed=hash((fl, sz)) % 1000)
        recs.append(dict(x=x, fault=fl, size=sz, rpm=1797., fs=12000))
    cfg = calibrate(recs, base=SimConfig(duration=10))
    print("\nTRUE vs RECOVERED")
    for k in ("size_exponent", "noise_std", "coloured_noise_std", "shaft_harm_amp",
              "jitter", "fault_jitter", "fault_modes"):
        print(f"{k:20s} true={getattr(truth, k)}\n{'':20s} fit ={getattr(cfg, k)}")
    report(cfg, recs)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--dataset", choices=["cwru", "spectraquest"], default="cwru")
    ap.add_argument("--axis", default="all", help="spectraquest: 0, 1, 2 or all")
    ap.add_argument("--out", default="calibrated.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no_realism", action="store_true", help="original stationary calibration (v1)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.dataset == "spectraquest":
        axes = [0, 1, 2] if a.axis == "all" else [int(a.axis)]
        base = SimConfig(fs=25600, duration=10, size_ref_in=0.001, double_impulse_size_in=1e9)
        result = {"_meta": dict(bearing="Rexnord ER-12K", fs=25600, rpm=1200,
                                severity="L/M/H stored as size 0.001/0.002/0.003")}
        for ax in axes:
            recs = load_spectraquest_dir(a.data_dir, ax)
            recs.sort(key=lambda r: (r["fault"], r["size"]))
            print(f"\n===== axis {ax + 1}: {len(recs)} files =====", flush=True)
            cfg = calibrate(recs, ER12K, base, band="auto", fmin=800, zmax=0.2, realism=not a.no_realism)
            result[f"Axis {ax + 1}"] = asdict(cfg)
            json.dump(result, open(a.out, "w"), indent=1, default=list)
            report(cfg, recs, ER12K)
            print(flush=True)
        print("saved", a.out)
    else:
        recs = load_cwru_dir(a.data_dir)
        recs.sort(key=lambda r: (r["fault"], r["size"]))
        print(f"loaded {len(recs)} files:", sorted(r["file"] for r in recs))
        cfg = calibrate(recs, realism=not a.no_realism)
        json.dump(asdict(cfg), open(a.out, "w"), indent=2, default=list)
        print("saved", a.out)
        report(cfg, recs)
