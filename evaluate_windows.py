"""
Window-level evaluation of calibrated configs against real recordings.

  python evaluate_windows.py --dataset cwru --data_dir DIR --split second --config a.json b.json

Scores (lower is better, 0 = perfect), averaged over cases and 3 simulation seeds:
  kurtosis / crest : |ln(sim/real)| of the MEDIAN per-window value
  rms-cv           : |difference| of the window-to-window RMS coefficient of variation
  env-peak         : |ln(sim/real)| of the fault peak in the window-averaged envelope spectrum
  spectrum [dB]    : RMS difference of the window-averaged log-PSD, 400 Hz .. 0.95 Nyquist
and, for CWRU, a train-on-synthetic / test-on-real classifier score (TSTR).
"""
import argparse
import json
from dataclasses import replace

import numpy as np
from scipy import signal, stats

from bearing_sim import CWRU_DE, ER12K, SimConfig, config_from_dict as cfg_from_dict, simulate
from calibrate import load_cwru_dir, load_spectraquest_dir
from calibrate_v3 import fault_rate, split_records, window_stats


def _old_cfg_from_dict(d):
    d = dict(d)
    tup = lambda m: tuple(tuple(x) for x in m)
    d["modes"] = tup(d["modes"])
    d["fault_modes"] = {k: tup(v) for k, v in d.get("fault_modes", {}).items()}
    d["case_modes"] = {k: tup(v) for k, v in d.get("case_modes", {}).items()}
    d["case_eq"] = {k: (tuple(v[0]), tuple(v[1])) for k, v in d.get("case_eq", {}).items()}
    for k in ("shaft_harm_amp", "bg_shape_f", "bg_shape_db"):
        if k in d:
            d[k] = tuple(d[k])
    if "bg_lines" in d:
        d["bg_lines"] = tuple(tuple(x) for x in d["bg_lines"])
    return SimConfig(**d)


def evaluate(cfg, records, bearing, win, seeds=(7, 8, 9), nmax=60):
    rows = {k: [] for k in ("kurtosis", "crest", "rms-cv", "env-peak", "spectrum [dB]")}
    healthy = {k: [] for k in ("kurtosis", "crest", "spectrum [dB]")}
    for r in records:
        fs, kind = r["fs"], r["fault"]
        f0 = None if kind == "normal" else fault_rate(bearing, r["rpm"], kind)
        band = tuple(cfg.env_band.get(kind, (2000, 5000)))
        sr = window_stats(r["x"], fs, win, f0, band, nmax)
        for sd in seeds:
            x, _ = simulate(bearing, kind, r["size"], r["rpm"], replace(cfg, duration=sr["n"] * win / fs), seed=sd)
            ss = window_stats(x, fs, win, f0, band, sr["n"])
            row = {"kurtosis": abs(ss["lk"] - sr["lk"]), "crest": abs(ss["lc"] - sr["lc"]),
                   "rms-cv": abs(ss["cv"] - sr["cv"]),
                   "spectrum [dB]": float(np.sqrt(np.mean((ss["psd"] - sr["psd"]) ** 2)))}
            if f0:
                row["env-peak"] = abs(ss["lp1"] - sr["lp1"])
            for k, v in row.items():
                rows[k].append(v)
                if kind == "normal" and k in healthy:
                    healthy[k].append(v)
    return ({k: float(np.mean(v)) for k, v in rows.items() if v},
            {k: float(np.mean(v)) for k, v in healthy.items() if v})


# ---------------------------------------------------------------- TSTR (CWRU)
def _tstr_features(W, fs, f0, band=(2000, 5000)):
    n = W.shape[1]
    Wc = W - W.mean(1, keepdims=True)
    r = Wc.std(1)
    P = np.abs(np.fft.rfft(Wc * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    bp = [np.log(P[:, (f >= a) & (f < b)].sum(1) + 1e-12) for a, b in [(0, 500), (500, 1500), (1500, 3000), (3000, 4500), (4500, 6000)]]
    e = np.abs(signal.hilbert(signal.sosfiltfilt(signal.butter(4, band, "band", fs=fs, output="sos"), Wc, axis=1), axis=1))
    e -= e.mean(1, keepdims=True)
    E = np.abs(np.fft.rfft(e * np.hanning(n), axis=1))
    fe = np.fft.rfftfreq(n, 1 / fs)
    fl = np.median(E[:, (fe > 20) & (fe < 600)], axis=1) + 1e-12
    pk = [np.log(E[:, (fe > f0[k] * 0.96) & (fe < f0[k] * 1.04)].max(1) / fl) for k in ("outer", "inner", "ball")]
    return np.column_stack([np.log(r), stats.kurtosis(Wc, axis=1, fisher=False), np.abs(Wc).max(1) / r,
                            stats.skew(Wc, axis=1)] + bp + pk)


def tstr(cfg, real_records, win=2048, records_per_case=8, windows_per_record=8, reps=5, return_pred=False):
    """Train on synthetic, test on real. Averaged over `reps` independently generated training sets."""
    from sklearn.ensemble import RandomForestClassifier
    fs = real_records[0]["fs"]
    f0 = {k: fault_rate(CWRU_DE, real_records[0]["rpm"], k) for k in ("outer", "inner", "ball")}
    cases = [("normal", 0.0)] + [(k, s) for k in ("inner", "ball", "outer") for s in (0.007, 0.014, 0.021)]
    dur = windows_per_record * win / fs
    Xr, yr, fid = [], [], []
    for r in real_records:
        m = len(r["x"]) // win
        Xr.append(_tstr_features(r["x"][: m * win].reshape(m, win), fs, f0))
        yr += [r["fault"]] * m
        fid += [r.get("file")] * m
    Xr, yr, fid = np.concatenate(Xr), np.array(yr), np.array(fid, dtype=object)
    oks, preds = [], []
    for rep in range(reps):
        Xs, ys = [], []
        for ci, (kind, sz) in enumerate(cases):
            for rec in range(records_per_case):
                x, _ = simulate(CWRU_DE, kind, sz, 1797.0, replace(cfg, duration=dur), seed=100000 * rep + 1000 * ci + rec)
                Xs.append(_tstr_features(x[: windows_per_record * win].reshape(windows_per_record, win), fs, f0))
                ys += [kind] * windows_per_record
        clf = RandomForestClassifier(300, random_state=rep).fit(np.concatenate(Xs), ys)
        pred = clf.predict(Xr)
        oks.append(pred == yr)
        preds.append(pred)
    ok = np.mean(oks, axis=0)
    mask = fid != 197                                    # file 197 is near-healthy in the real data
    per = {int(f): float(ok[fid == f].mean()) for f in sorted({x for x in fid})}
    if return_pred:
        return float(ok.mean()), float(ok[mask].mean()), per, np.concatenate(preds), np.tile(yr, reps), np.tile(fid, reps)
    return float(ok.mean()), float(ok[mask].mean()), per


def c2st(cfg, records, bearing=CWRU_DE, win=2048, nsim_records=4, seed=0):
    """Classifier two-sample test per case: can a random forest tell REAL windows from SYNTHETIC ones?
    5-fold CV accuracy; 0.5 = indistinguishable, 1.0 = trivially separable. Returns {case: accuracy}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    out = {}
    for r in records:
        fs, kind = r["fs"], r["fault"]
        f0 = {k: fault_rate(bearing, r["rpm"], k) for k in ("outer", "inner", "ball")}
        m = len(r["x"]) // win
        Wr = r["x"][: m * win].reshape(m, win)
        per = int(np.ceil(m / nsim_records))
        Ws = []
        for i in range(nsim_records):
            x, _ = simulate(bearing, kind, r["size"], r["rpm"], replace(cfg, duration=per * win / fs), seed=5000 + 10 * seed + i)
            Ws.append(x[: per * win].reshape(per, win))
        Ws = np.concatenate(Ws)[:m]
        X = np.concatenate([_tstr_features(Wr, fs, f0), _tstr_features(Ws, fs, f0)])
        y = np.r_[np.zeros(m), np.ones(len(Ws))]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        out[f"{kind} {r['size']}"] = float(cross_val_score(RandomForestClassifier(200, random_state=0), X, y, cv=cv).mean())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cwru", "spectraquest"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--split", choices=["full", "first", "second"], default="full")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--tstr", action="store_true")
    a = ap.parse_args()
    if a.dataset == "cwru":
        recs, bearing, win = split_records(load_cwru_dir(a.data_dir), a.split), CWRU_DE, 2048
    else:
        recs, bearing, win = split_records(load_spectraquest_dir(a.data_dir, a.axis - 1, seconds=15.0), a.split), ER12K, 4096
    cols = ["kurtosis", "crest", "rms-cv", "env-peak", "spectrum [dB]"]
    print(f"real portion: {a.split}; {len(recs)} files. Lower is better.\n")
    print(f"{'config':<44}" + "".join(f"{c:>14}" for c in cols) + ("      TSTR   (excl. file 197)" if a.tstr else ""))
    for path in a.config:
        d = json.load(open(path))
        if a.dataset == "spectraquest":
            d = d[f"Axis {a.axis}"]
        cfg = cfg_from_dict(d)
        tot, hl = evaluate(cfg, recs, bearing, win)
        line = f"{path.split('/')[-1]:<44}" + "".join(f"{tot.get(c, float('nan')):>14.2f}" for c in cols)
        if a.tstr and a.dataset == "cwru":
            acc, acc2, per = tstr(cfg, recs)
            line += f"   {acc:6.3f}   {acc2:6.3f}"
        print(line, flush=True)
        if a.tstr and a.dataset == "cwru":
            cs = c2st(cfg, recs)
            print(f"{'   real-vs-synthetic test (0.5 = best)':<44} mean {np.mean(list(cs.values())):.3f}   " + "  ".join(f"{k.split()[0][:2]}{k.split()[1][2:]}:{v:.2f}" for k, v in cs.items()))
        print(f"{'   healthy file only':<44}" + "".join(f"{hl.get(c, float('nan')):>14.2f}" if c in hl else f"{'':>14}" for c in cols))
