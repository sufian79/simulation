"""
Calibration v3: make synthetic WINDOWS statistically close to real windows.

New compared with calibrate.py (v2):
  1. recorder anti-aliasing filter detected from the data (CWRU: steep cliff near 5.4 kHz; none for SpectraQuest)
  2. background noise shaped to the real healthy recording's PSD (instead of white + 1.5 kHz low-pass)
  3. fault parameters (slip, impact scatter, non-impulsive share, amplitude drift) are optimised against
     WINDOW-level statistics: median kurtosis, median crest, window-to-window RMS variability,
     averaged envelope peaks and the averaged log-PSD  (v2 used whole-file statistics)
  4. optional --split first/second: fit on one half of every file so the other half can be used as a
     genuinely unseen test (see evaluate_windows.py)

  python calibrate_v3.py --dataset cwru --data_dir DIR --out calibrated_cwru.json
  python calibrate_v3.py --dataset cwru --data_dir DIR --split first --out /tmp/cwru_first.json
  python calibrate_v3.py --dataset spectraquest --data_dir DIR --axis all --out calibrated_spectraquest.json
"""
import argparse
import itertools
import json
import time
from dataclasses import asdict, replace

import numpy as np
from scipy import signal, stats
from scipy.ndimage import median_filter, uniform_filter1d

from bearing_sim import CWRU_DE, ER12K, SimConfig, simulate
from calibrate import (fit_amplitudes, fit_background, fit_modes, fit_modes_psd,
                       fit_noise_and_harmonics, load_cwru_dir, load_spectraquest_dir, trim_gains)

_BAND_CACHE = {}


def _band_sos(band, fs):
    key = (tuple(band), fs)
    if key not in _BAND_CACHE:
        _BAND_CACHE[key] = signal.butter(4, band, btype="band", fs=fs, output="sos")
    return _BAND_CACHE[key]


def split_records(records, part):
    """part: 'full' | 'first' | 'second' -> that portion of every recording."""
    if part == "full":
        return records
    out = []
    for r in records:
        h = len(r["x"]) // 2
        out.append({**r, "x": r["x"][:h] if part == "first" else r["x"][h:]})
    return out


def fault_rate(bearing, rpm, kind):
    fq = bearing.freqs(rpm / 60.0)
    return {"outer": fq["BPFO"], "inner": fq["BPFI"], "ball": 2 * fq["BSF"]}.get(kind)


# ----------------------------------------------------------------- window statistics
def window_stats(x, fs, win, f0, band, nmax=None):
    m = len(x) // win
    if nmax:
        m = min(m, nmax)
    W = x[: m * win].reshape(m, win)
    Wc = W - W.mean(1, keepdims=True)
    r = Wc.std(1)
    out = dict(lk=float(np.log(np.median(stats.kurtosis(Wc, axis=1, fisher=False)))),
               lc=float(np.log(np.median(np.abs(Wc).max(1) / r))),
               cv=float(r.std() / r.mean()), n=m)
    f, P = signal.welch(Wc, fs, nperseg=win // 2, axis=-1)
    sel = (f > 400) & (f < 0.95 * fs / 2)
    out["psd"] = 10 * np.log10(P.mean(0)[sel] + 1e-20)
    if f0:
        e = np.abs(signal.hilbert(signal.sosfiltfilt(_band_sos(band, fs), Wc, axis=1), axis=1))
        e -= e.mean(1, keepdims=True)
        S = np.abs(np.fft.rfft(e * np.hanning(win), axis=1)).mean(0)
        fe = np.fft.rfftfreq(win, 1 / fs)
        fl = np.median(S[(fe > 20) & (fe < 600)])
        df_ = fe[1]

        def pk(g):                                   # +-4 % band; widened if the resolution is too coarse
            sel_ = (fe > g * 0.96) & (fe < g * 1.04)
            if not sel_.any():
                sel_ = np.abs(fe - g) <= 1.5 * df_
            return S[sel_].max() / fl
        p1, p2 = pk(f0), pk(2 * f0)
        out["lp1"], out["lr"] = float(np.log(p1)), float(np.log(p2 / p1))
    return out


def stats_cost(sr, ss):
    d = [ss["lk"] - sr["lk"], ss["lc"] - sr["lc"], 3.0 * (ss["cv"] - sr["cv"])]
    if "lp1" in sr:
        d += [ss["lp1"] - sr["lp1"], 0.5 * (ss["lr"] - sr["lr"])]
    d.append(float(np.sqrt(np.mean((ss["psd"] - sr["psd"]) ** 2))) / 8.0)
    return float(sum(v * v for v in d))


# ----------------------------------------------------------------- recorder / background
def fit_antialias(records):
    """Detect a steep anti-aliasing cliff in the pooled PSD; return (fc, order) or (0, 8)."""
    fs = records[0]["fs"]
    nyq = fs / 2
    rel = []
    for r in records:
        f, P = signal.welch(r["x"] - r["x"].mean(), fs, nperseg=4096)
        db = 10 * np.log10(P + 1e-20)
        rel.append(db - np.median(db[(f > 0.25 * nyq) & (f < 0.7 * nyq)]))
    D = uniform_filter1d(np.median(rel, axis=0), 5)
    hi = f > 0.7 * nyq
    idx = np.where(hi & (D < -20))[0]
    if len(idx) == 0:
        return 0.0, 8
    f_cliff = f[idx[0]]
    top = (f > f_cliff + 0.04 * nyq) & (f < 0.995 * nyq)
    if not top.any() or np.median(D[top]) > -35:       # not a steep cliff -> no recorder filter
        return 0.0, 8
    band = (f > 0.8 * nyq) & (f < 0.995 * nyq)
    flat = (f > 0.8 * nyq) & (f < f_cliff - 0.04 * nyq)
    best = (np.inf, 0.0, 8)
    for order in (6, 8, 10):
        for fc in np.arange(f_cliff - 450, f_cliff + 100, 25.0):
            sos = signal.ellip(order, 1.0, 80.0, fc / nyq, output="sos")
            _, h = signal.sosfreqz(sos, worN=f[band], fs=fs)
            model = np.maximum(20 * np.log10(np.abs(h) + 1e-12), -70)
            real = np.maximum(D[band], -70)
            fl = flat[band] if flat.sum() else np.ones(band.sum(), bool)
            off = np.median(real[fl] - model[fl]) if fl.any() else 0.0
            cost = float(np.mean((real - (off + model)) ** 2))
            best = min(best, (cost, float(fc), order))
    return best[1], best[2]


def fit_bg_lines(normal, thr_db=12.0, nmax=8):
    """Narrow spectral lines (above a robust baseline) in the healthy recording, excluding shaft 1-3x."""
    fs, fr = normal["fs"], normal["rpm"] / 60.0
    f, P = signal.welch(normal["x"] - normal["x"].mean(), fs, nperseg=8192)
    df = f[1] - f[0]
    db = 10 * np.log10(P + 1e-20)
    base = median_filter(db, size=61, mode="nearest")
    pk, _ = signal.find_peaks(db - base, height=thr_db, distance=3)
    out = []
    for i in pk:
        if f[i] < 20 or any(abs(f[i] - k * fr) < 2.0 for k in (1, 2, 3)):
            continue
        nb = max(int(12.0 / df), 3)                       # +-12 Hz: includes shaft-rate sidebands
        sl = slice(max(i - nb, 0), i + nb + 1)
        pw = float(np.clip(P[sl] - 10 ** (base[sl] / 10), 0, None).sum() * df)
        out.append((round(float(f[i]), 1), float(np.sqrt(2 * pw)), pw))
    out = sorted(out, key=lambda t: -t[2])[:nmax]
    return tuple((fq, round(a, 5)) for fq, a, _ in sorted(out))


def fit_bg_shape(normal, aa_fc, npts=96, lines=()):
    """Smooth PSD shape (shaft lines removed) of the healthy recording, flat beyond the AA edge."""
    fs = normal["fs"]
    nyq = fs / 2
    f, P = signal.welch(normal["x"] - normal["x"].mean(), fs, nperseg=4096, noverlap=2048)
    db = 10 * np.log10(P + 1e-20)
    if lines:                                  # interpolate the spectrum across the narrow lines
        keep = np.ones_like(f, bool)
        for fq, _ in lines:
            keep &= abs(f - fq) > 18.0
        db = np.interp(f, f[keep], db[keep])
    db = uniform_filter1d(median_filter(db, size=41, mode="nearest"), 15)
    edges = np.linspace(0, nyq, npts + 1)
    fc_pts = 0.5 * (edges[:-1] + edges[1:])
    lv = np.array([db[(f >= a) & (f < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    if aa_fc > 0:
        lv = np.where(fc_pts > 0.97 * aa_fc, lv[np.argmin(abs(fc_pts - 0.97 * aa_fc))], lv)
    lv = np.maximum(lv - lv.max(), -60.0)
    fgrid = np.concatenate([[0.0], fc_pts, [nyq]])
    lgrid = np.concatenate([[lv[0]], lv, [lv[-1]]])
    return tuple(round(float(v), 1) for v in fgrid), tuple(round(float(v), 2) for v in lgrid)


def trim_noise(cfg, normal, bearing, seeds=(3, 4, 5), dur=8.0):
    """Scale the background so the simulated healthy RMS equals the real healthy RMS."""
    fs = normal["fs"]
    d_ = min(dur, len(normal["x"]) / fs)
    vr = float(np.var(normal["x"][: int(d_ * fs)]))
    c0 = replace(cfg, duration=d_, noise_std=0.0, coloured_noise_std=0.0)
    cn = replace(cfg, duration=d_)
    v0 = np.mean([np.var(simulate(bearing, "normal", 0.0, normal["rpm"], c0, seed=s)[0]) for s in seeds])
    v1 = np.mean([np.var(simulate(bearing, "normal", 0.0, normal["rpm"], cn, seed=s)[0]) for s in seeds])
    k = float(np.sqrt(max(vr - v0, 0.05 * vr) / max(v1 - v0, 1e-12)))
    return replace(cfg, noise_std=cfg.noise_std * k, coloured_noise_std=cfg.coloured_noise_std * k)


def fit_case_modes(normal, faults, per_kind, ir_len_ms, split_hz=1500.0, fmax=None):
    """Per-case resonance modes: high band (> split_hz) fitted to that case's own excess PSD,
    low band (< split_hz) taken from the fault-type fit."""
    out = {}
    for r in faults:
        kind = r["fault"]
        hi, _ = fit_modes_psd(normal, [r], [m for m in per_kind[kind] if m[0] >= split_hz] or per_kind[kind],
                              K=4, fmin=split_hz, ir_len_ms=ir_len_ms, fmax=fmax)
        lo = [m for m in per_kind[kind] if m[0] < split_hz]
        gmax = max(m[2] for m in per_kind[kind])
        out[f"{kind}@{r['size']:.3f}"] = tuple(sorted(list(hi) + lo, key=lambda m: -m[2]))
    return out


def fit_eq(cfg, records, bearing, win, npts=64, nmax=48, seeds=(11, 12, 13), damp=0.85, clip=15.0):
    """Smooth, power-preserving spectral correction per case so the window-averaged PSD of the simulated
    signal follows the real one (the structure-to-sensor transfer function the modes cannot capture)."""
    fs = records[0]["fs"]
    nyq = fs / 2
    fmax = 0.95 * cfg.aa_fc if cfg.aa_fc else 0.95 * nyq
    pts = np.geomspace(60.0, fmax, npts)
    lo_edge, hi_edge = pts * 0.93, pts * 1.07
    eq = dict(cfg.case_eq)
    for r in records:
        kind = r["fault"]
        k = f"{kind}@{r['size']:.3f}" if kind != "normal" else "normal@0.000"
        m = min(len(r["x"]) // win, nmax)
        Wr = r["x"][: m * win].reshape(m, win)
        f, Pr = signal.welch(Wr - Wr.mean(1, keepdims=True), fs, nperseg=1024, axis=-1)
        Pr = Pr.mean(0)
        cc = replace(cfg, duration=m * win / fs, case_eq=eq)
        Ps = 0
        for sd in seeds:
            x, _ = simulate(bearing, kind, r["size"], r["rpm"], cc, seed=sd)
            W = x[: m * win].reshape(m, win)
            Ps = Ps + signal.welch(W - W.mean(1, keepdims=True), fs, nperseg=1024, axis=-1)[1].mean(0) / len(seeds)
        ratio = 10 * np.log10(Pr + 1e-20) - 10 * np.log10(Ps + 1e-20)
        delta = np.array([np.mean(ratio[(f >= a) & (f < b)]) if ((f >= a) & (f < b)).any() else 0.0
                          for a, b in zip(lo_edge, hi_edge)])
        delta = np.clip(damp * delta, -clip, clip)
        old = np.interp(pts, *eq[k]) if k in eq else np.zeros(npts)
        new = old + delta
        # power-preserving: renormalise so the total power implied by the correction is unchanged
        g_new = np.interp(f, pts, new, left=new[0], right=new[-1]) - np.interp(f, pts, old, left=old[0], right=old[-1])
        rho = float(np.sum(Ps * 10 ** (g_new / 10)) / np.sum(Ps))
        new = new - 10 * np.log10(rho)
        fgrid = np.concatenate([[0.0], pts, [nyq]])
        dgrid = np.concatenate([[new[0]], new, [new[-1]]])
        eq[k] = (tuple(round(float(v), 1) for v in fgrid), tuple(round(float(v), 2) for v in dgrid))
    return replace(cfg, case_eq=eq)


# ----------------------------------------------------------------- window-level fit
def fit_window_level(cfg, records, bearing, bands, win, rounds=2, nmax=48, seeds=(1, 2)):
    faults = [r for r in records if r["fault"] != "normal"]
    kinds = sorted({r["fault"] for r in faults})
    real = {}
    for r in faults:
        f0 = fault_rate(bearing, r["rpm"], r["fault"])
        real[id(r)] = window_stats(r["x"], r["fs"], win, f0, bands[r["fault"]], nmax)

    def case_cost(c, r):
        m = real[id(r)]["n"]
        cc = replace(c, duration=m * win / r["fs"])
        f0 = fault_rate(bearing, r["rpm"], r["fault"])
        tot = 0.0
        for sd in seeds:
            x, _ = simulate(bearing, r["fault"], r["size"], r["rpm"], cc, seed=sd)
            tot += stats_cost(real[id(r)], window_stats(x, r["fs"], win, f0, bands[r["fault"]], m))
        return tot / len(seeds)

    key = lambda r: f"{r['fault']}@{r['size']:.3f}"

    def fit_bg(cfg, r):
        """Background level of this fault recording relative to the healthy one; the impact amplitude is
        re-solved analytically so the case RMS stays equal to the real RMS."""
        k_ = key(r)
        m = real[id(r)]["n"]
        cc = replace(cfg, duration=m * win / r["fs"])
        c0 = replace(cc, case_gain={**cfg.case_gain, k_: 0.0})
        vt = np.mean([np.var(simulate(bearing, r["fault"], r["size"], r["rpm"], cc, seed=s)[0]) for s in (21, 22)])
        vn1 = np.mean([np.var(simulate(bearing, r["fault"], r["size"], r["rpm"], c0, seed=s)[0]) for s in (21, 22)])
        vr = float(np.var(r["x"][: m * win]))
        s0 = cfg.case_bg.get(k_, 1.0)
        vi = max(vt - vn1, 1e-12)
        best = (np.inf, s0, cfg.case_gain[k_])
        for sc in BG:
            vn = vn1 * (sc / s0) ** 2
            g = cfg.case_gain[k_] * float(np.sqrt(max(vr - vn, 0.1 * vr) / vi))
            c2 = replace(cfg, case_bg={**cfg.case_bg, k_: sc}, case_gain={**cfg.case_gain, k_: g})
            best = min(best, (case_cost(c2, r), sc, g))
        return replace(cfg, case_bg={**cfg.case_bg, k_: best[1]}, case_gain={**cfg.case_gain, k_: best[2]})

    BG = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
    J = (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3)
    SC = (0.05, 0.15, 0.3, 0.5)
    G = (0.0, 0.3, 0.6, 0.8, 0.92, 0.97, 0.99)
    DR = (0.0, 0.5, 1.0, 1.5)
    KP = (0.0, 0.5, 1.5, 3.0, 5.0, 8.0, 12.0)
    MD = (0.0, 0.3, 0.6, 0.9)
    DT = (0.03, 0.1, 0.5)
    for rd in range(rounds):
        t0 = time.time()
        for r in faults:                                     # every parameter is fitted per case
            k = key(r)
            for name, grid in (("case_jitter", J), ("case_scatter", SC)):
                best = min((case_cost(replace(cfg, **{name: {**getattr(cfg, name), k: v}}), r), v) for v in grid)
                cfg = replace(cfg, **{name: {**getattr(cfg, name), k: best[1]}})
            best = min((case_cost(replace(cfg, case_gfrac={**cfg.case_gfrac, k: g},
                                          case_kappa={**cfg.case_kappa, k: kp}), r), g, kp) for g in G for kp in KP)
            cfg = replace(cfg, case_gfrac={**cfg.case_gfrac, k: best[1]}, case_kappa={**cfg.case_kappa, k: best[2]})
            cfg = fit_bg(cfg, r)
            if r["fault"] in ("inner", "ball"):               # load-zone / cage modulation depth
                best = min((case_cost(replace(cfg, case_mod={**cfg.case_mod, k: md}), r), md) for md in MD)
                cfg = replace(cfg, case_mod={**cfg.case_mod, k: best[1]})
            best = min((case_cost(replace(cfg, case_drift={**cfg.case_drift, k: d}, case_drift_tau={**cfg.case_drift_tau, k: dt}), r), d, dt)
                       for d in DR for dt in DT)
            cfg = replace(cfg, case_drift={**cfg.case_drift, k: best[1]}, case_drift_tau={**cfg.case_drift_tau, k: best[2]})
        cfg = trim_gains(cfg, records, bearing, iters=2)
        tot = sum(case_cost(cfg, r) for r in faults) / len(faults)
        print(f"  round {rd + 1}: mean window-level cost {tot:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    print("  jitter:", cfg.case_jitter, flush=True)
    print("  scatter:", cfg.case_scatter, flush=True)
    print("  kappa:", cfg.case_kappa, flush=True)
    print("  mod:", cfg.case_mod, flush=True)
    print("  drift tau:", cfg.case_drift_tau, flush=True)
    print("  gfrac:", {k: v for k, v in cfg.case_gfrac.items()}, flush=True)
    print("  drift:", {k: v for k, v in cfg.case_drift.items()}, flush=True)
    print("  bg level:", {k: v for k, v in cfg.case_bg.items()}, flush=True)
    return cfg


# ----------------------------------------------------------------- driver
def calibrate_v3(records, bearing, base, band, fmin, zmax, win, rounds=2):
    normal = next(r for r in records if r["fault"] == "normal")
    faults = [r for r in records if r["fault"] != "normal"]
    kinds = sorted({r["fault"] for r in faults})
    fs = normal["fs"]
    cfg = replace(base, fs=fs)
    cfg = replace(cfg, **fit_noise_and_harmonics(normal))
    pooled = fit_modes(normal, faults, fmin=fmin, zmax=zmax)
    per_kind = {k: fit_modes(normal, [r for r in faults if r["fault"] == k], fmin=fmin, zmax=zmax) for k in kinds}
    aa_fc, aa_order = fit_antialias(records)
    fmax = 0.88 * aa_fc if aa_fc else None                # keep the recorder roll-off out of the mode fit
    pooled, _ = fit_modes_psd(normal, faults, pooled, fmin=fmin, ir_len_ms=cfg.ir_len_ms, fmax=fmax)
    for k in kinds:
        per_kind[k], (e0, e1) = fit_modes_psd(normal, [r for r in faults if r["fault"] == k], per_kind[k],
                                              fmin=fmin, ir_len_ms=cfg.ir_len_ms, fmax=fmax)
        print(f"spectral fit {k:6s}: {e0:.1f} dB -> {e1:.1f} dB", flush=True)
    cfg = replace(cfg, modes=pooled, fault_modes=per_kind)
    if band == "auto":
        bands = {k: (int(max(300, 0.7 * per_kind[k][0][0])), int(min(0.48 * fs, 1.35 * per_kind[k][0][0]))) for k in kinds}
    else:
        bands = {k: tuple(band) for k in kinds}
    cfg = replace(cfg, env_band={k: list(v) for k, v in bands.items()})
    # recorder + background
    lines = fit_bg_lines(normal)
    print("healthy-recording lines (Hz, amp):", lines, flush=True)
    bf, bdb = fit_bg_shape(normal, aa_fc, lines=lines)
    nrms = float(np.sqrt(cfg.noise_std ** 2 + cfg.coloured_noise_std ** 2))
    cfg = replace(cfg, aa_fc=aa_fc, aa_order=aa_order, bg_shape_f=bf, bg_shape_db=bdb, bg_lines=lines,
                  noise_std=nrms, coloured_noise_std=0.0)
    print(f"recorder anti-aliasing: {'fc = %.0f Hz, order %d' % (aa_fc, aa_order) if aa_fc else 'none detected'}", flush=True)
    cfg = trim_noise(cfg, normal, bearing)
    cfg = fit_background(normal, cfg, bearing)               # transients / tails for the healthy file
    cfg = trim_noise(cfg, normal, bearing)
    cm = fit_case_modes(normal, faults, per_kind, cfg.ir_len_ms, fmax=fmax)
    cfg = replace(cfg, case_modes=cm)
    print("per-case modes:", {k: [m[0] for m in v] for k, v in cm.items()}, flush=True)
    # amplitudes and starting values
    case, p, gains, r2 = fit_amplitudes(cfg, normal, faults, bearing)
    start = {f"{r['fault']}@{r['size']:.3f}": 0.6 for r in faults}
    cfg = replace(cfg, case_gain=case, size_exponent=p, a0=1.0, fault_gain=gains, case_gfrac=start,
                  case_drift={k: 0.0 for k in start}, case_bg={k: 1.0 for k in start},
                  case_jitter={k: 0.02 for k in start}, case_scatter={k: 0.15 for k in start},
                  case_kappa={k: 3.0 for k in start},
                  case_mod={k: 0.8 for k in start}, case_drift_tau={k: 0.5 for k in start}, fault_jitter={k: 0.02 for k in kinds},
                  fault_scatter={k: 0.15 for k in kinds})
    cfg = trim_gains(cfg, records, bearing, iters=2)
    cfg = fit_window_level(cfg, records, bearing, bands, win, rounds=max(rounds - 1, 1))
    cfg = fit_eq(cfg, records, bearing, win)
    cfg = fit_window_level(cfg, records, bearing, bands, win, rounds=1)
    cfg = fit_eq(cfg, records, bearing, win)
    cfg = trim_gains(cfg, records, bearing, iters=2)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cwru", "spectraquest"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--axis", default="all")
    ap.add_argument("--split", choices=["full", "first", "second"], default="full")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.dataset == "cwru":
        recs = split_records(load_cwru_dir(a.data_dir), a.split)
        recs.sort(key=lambda r: (r["fault"], r["size"]))
        print(f"CWRU: {len(recs)} files, split={a.split}", flush=True)
        cfg = calibrate_v3(recs, CWRU_DE, SimConfig(duration=10), (2000, 5000), 400, 0.12, 2048, a.rounds)
        json.dump(asdict(cfg), open(a.out, "w"), indent=1, default=list)
    else:
        axes = [0, 1, 2] if a.axis == "all" else [int(a.axis) - 1]
        base = SimConfig(fs=25600, duration=10, size_ref_in=0.001, double_impulse_size_in=1e9)
        result = {"_meta": dict(bearing="Rexnord ER-12K", fs=25600, rpm=1200, version=3,
                                severity="L/M/H stored as size 0.001/0.002/0.003")}
        for ax in axes:
            recs = split_records(load_spectraquest_dir(a.data_dir, ax, seconds=15.0), a.split)
            recs.sort(key=lambda r: (r["fault"], r["size"]))
            print(f"\n===== SpectraQuest axis {ax + 1} =====", flush=True)
            result[f"Axis {ax + 1}"] = asdict(calibrate_v3(recs, ER12K, base, "auto", 800, 0.2, 4096, a.rounds))
            json.dump(result, open(a.out, "w"), indent=1, default=list)
    print("saved", a.out)
