"""(A) Validation of R6 RSI fade on RTY: bar size, RSI length, thresholds, exit level, time stop; walk-forward
2017-07..2021 (train) vs 2022..2026-10 (test); cost 2/4/6 ticks; per year; same best rule on ES / YM / NQ.
(B) R9 first-hour momentum (|open -> +60 min move| > k sd -> hold to close) on NQ / RTY / ES / YM with filters and
exit variants one at a time. Net of 4 ticks, bps, per-day P&L."""
import itertools

import numpy as np
import pandas as pd

import lit_futures_grid as g
from research4 import GROUPS
from research8 import COST
from config import DATA_DIR

D = DATA_DIR
pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 400)


def bars(d, size):
    cols = np.arange(size, d.M + 1, size)
    return cols, d.C[:, cols]


def rsi_series(c, n):
    s = pd.Series(c.reshape(-1))
    dl = s.diff()
    gn = dl.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ls = (-dl.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + gn / ls.replace(0, np.nan))).to_numpy().reshape(c.shape)


def rsi_fade(d, size, n, lo_th, exit_lvl, max_bars, cost):
    cols, c = bars(d, size)
    r = rsi_series(c, n)
    nd, nb = c.shape
    P = np.zeros((nd, nb))
    pos, age = np.zeros(nd), np.zeros(nd)
    for b in range(nb):
        age = np.where(pos != 0, age + 1, 0)
        ex = ((pos == 1) & (r[:, b] >= exit_lvl)) | ((pos == -1) & (r[:, b] <= 100 - exit_lvl))
        if max_bars:
            ex |= age >= max_bars
        pos = np.where(ex, 0, pos)
        el, es = (pos == 0) & (r[:, b] < lo_th), (pos == 0) & (r[:, b] > 100 - lo_th)
        pos = np.where(el, 1, np.where(es, -1, pos))
        age = np.where(el | es, 0, age)
        P[:, b] = pos
    P[:, -1] = 0
    px = np.concatenate([d.C[:, :1], c], 1)
    held = np.concatenate([np.zeros((nd, 1)), P], 1)
    pnl = (held[:, :-1] * np.diff(px, axis=1)).sum(1)
    chg = np.abs(np.diff(np.concatenate([np.zeros((nd, 1)), P], 1), axis=1)).sum(1)
    n_tr = ((np.abs(P) > 0) & (np.concatenate([np.zeros((nd, 1)), P[:, :-1]], 1) != P)).sum(1)
    bps = (pnl - chg * cost / 2) / d.C[:, 0] * 1e4
    return pd.DataFrame({"date": d.dates, "bps": bps, "n": n_tr})


def stats(x, periods):
    out = {}
    for p, lo, hi in periods:
        y = x[(x.date >= lo) & (x.date < hi)]
        if y.n.sum() < 30:
            continue
        t = y[y.n > 0]
        out[f"pf {p}"] = t.bps[t.bps > 0].sum() / -t.bps[t.bps < 0].sum()
        out[f"sh {p}"] = y.bps.mean() / y.bps.std() * np.sqrt(252)
        out[f"trades {p}"] = int(y.n.sum())
    return out


def part_a():
    d = g.Days("RTY")
    WF = (("train 2017-21", "2017-01-01", "2022-01-01"), ("test 2022-26", "2022-01-01", "2027-01-01"))
    rows = []
    for size, n, lo_th, ex, mb in itertools.product((5, 15, 30), (7, 14, 21), (20, 25, 30), (50, 60), (4, 8, 0)):
        mbars = 0 if mb == 0 else max(1, int(mb * 15 / size))  # time stop expressed in 15-min units: 1h / 2h / none
        x = rsi_fade(d, size, n, lo_th, ex, mbars, COST["RTY"])
        rows.append(dict(bar_min=size, rsi_len=n, entry=f"<{lo_th}/>{100 - lo_th}", exit_rsi=ex,
                         time_stop={4: "1h", 8: "2h", 0: "none"}[mb], **stats(x, WF)))
    res = pd.DataFrame(rows)
    res.to_csv(D + "validate_r6_rty_sweep.csv", index=False)
    both = (res["pf train 2017-21"] > 1) & (res["pf test 2022-26"] > 1)
    rho = res[["sh train 2017-21", "sh test 2022-26"]].corr(method="spearman").iloc[0, 1]
    print(f"=== A. RTY RSI fade sweep: {both.sum()}/{len(res)} configs positive in both halves; train->test Sharpe rank corr {rho:.2f}")
    base = res[(res.bar_min == 15) & (res.rsi_len == 14) & (res.entry == "<25/>75") & (res.exit_rsi == 50) & (res.time_stop == "2h")]
    print("original config:\n", base.round(2).to_string(index=False))
    print("\ntop 10 by TRAIN Sharpe -> test:")
    print(res.sort_values("sh train 2017-21", ascending=False).head(10).round(2).to_string(index=False))
    print("\naverage test PF by parameter:")
    for c in ("bar_min", "rsi_len", "entry", "exit_rsi", "time_stop"):
        print(res.groupby(c)[["pf train 2017-21", "pf test 2022-26"]].mean().round(3).to_string(), "\n")
    x = rsi_fade(d, 15, 14, 25, 50, 8, COST["RTY"])
    print("original per year (net bps, trades):")
    print(x.groupby(x.date.dt.year).agg(bps=("bps", "sum"), trades=("n", "sum")).round(0).T.to_string())
    for ticks in (2, 6):
        xx = rsi_fade(d, 15, 14, 25, 50, 8, ticks * 0.1)
        print(f"original @ {ticks} ticks:", {k: round(v, 2) for k, v in stats(xx, WF).items()})
    P3 = (("2010-14", "2010-01-01", "2015-01-01"), ("2015-22", "2015-01-01", "2023-01-01"), ("2023-26", "2023-01-01", "2027-01-01"))
    for mk in ("ES", "YM", "NQ"):
        xs = pd.concat([rsi_fade(dd, 15, 14, 25, 50, 8, COST[mk]) for dd in (g.Days(a) for a in GROUPS[mk])])
        print(f"same rule on {mk}:", {k: round(v, 2) for k, v in stats(xs, P3).items()})


# ------------------------------------------------------------------ B
def r9(d, cost, win=60, k=0.5, gap_dir=None, skip_extreme=False, vol=None, wd_skip=None, range_filter=None,
       vwap_side=False, stop=None, exit_hhmm=None, prior_day=None):
    C, H, L, O, M, VW = d.C, d.H, d.L, d.O, d.M, d.vwap
    r = C[:, win] / C[:, 0] - 1
    sd = pd.Series(r).rolling(60, min_periods=30).std().shift(1).to_numpy()
    rr = C[:, -1] / d.prev_close - 1
    rsd = pd.Series(rr).rolling(60, min_periods=30).std().shift(1).to_numpy()
    prior_z = np.r_[np.nan, (rr / rsd)[:-1]]
    rv = pd.Series(rr).rolling(14, min_periods=10).std().shift(1)
    rvp = rv.rolling(253, min_periods=61).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).to_numpy()
    rng1 = H[:, 1:win + 1].max(1) - L[:, 1:win + 1].min(1)
    rng_avg = pd.Series(rng1).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    gap = C[:, 0] - d.prev_close
    te = M if exit_hhmm is None else d.col(exit_hhmm)
    out = []
    for i in range(len(C)):
        if np.isnan(sd[i]) or abs(r[i]) <= k * sd[i]:
            continue
        s = np.sign(r[i])
        if gap_dir == "with" and np.sign(gap[i]) != s:
            continue
        if gap_dir == "against" and np.sign(gap[i]) != -s:
            continue
        if skip_extreme and abs(prior_z[i]) > 2:
            continue
        if vol == "low" and not rvp[i] < 0.5:
            continue
        if vol == "high" and not rvp[i] >= 0.5:
            continue
        if wd_skip is not None and d.dates[i].weekday() == wd_skip:
            continue
        if range_filter == "wide" and not rng1[i] > rng_avg[i]:
            continue
        if range_filter == "narrow" and not rng1[i] <= rng_avg[i]:
            continue
        if vwap_side and np.sign(C[i, win] - VW[i, win]) != s:
            continue
        if prior_day == "same" and np.sign(prior_z[i]) != s:
            continue
        if prior_day == "opposite" and np.sign(prior_z[i]) != -s:
            continue
        p = C[i, win]
        px = C[i, te]
        if stop is not None:
            st = {"open": C[i, 0], "mid": (H[i, 1:win + 1].max() + L[i, 1:win + 1].min()) / 2,
                  "first-hour extreme": L[i, 1:win + 1].min() if s == 1 else H[i, 1:win + 1].max()}[stop]
            if (st - p) * s < 0:
                hit = (L[i, win + 1:te + 1] <= st) if s == 1 else (H[i, win + 1:te + 1] >= st)
                if hit.any():
                    j = win + 1 + hit.argmax()
                    px = min(st, O[i, j]) if s == 1 else max(st, O[i, j])
        out.append((d.dates[i], (s * (px - p) - cost) / p * 1e4, 1))
    return pd.DataFrame(out, columns=["date", "bps", "n"])


def part_b():
    P3 = (("2010-14", "2010-01-01", "2015-01-01"), ("2015-22", "2015-01-01", "2023-01-01"), ("2023-26", "2023-01-01", "2027-01-01"))
    variants = {
        "base (60 min, k=0.5, hold to close)": {},
        "first 30 min": dict(win=30), "first 90 min": dict(win=90),
        "k = 0.25": dict(k=0.25), "k = 1.0": dict(k=1.0),
        "gap direction only": dict(gap_dir="with"), "against the gap": dict(gap_dir="against"),
        "skip after |prior day| > 2 sd (H19)": dict(skip_extreme=True),
        "low-vol regime": dict(vol="low"), "high-vol regime": dict(vol="high"),
        "skip Mondays": dict(wd_skip=0), "skip Fridays": dict(wd_skip=4),
        "wide first hour (> 20-day avg)": dict(range_filter="wide"), "narrow first hour": dict(range_filter="narrow"),
        "price on the VWAP side": dict(vwap_side=True),
        "same direction as prior day": dict(prior_day="same"), "opposite to prior day": dict(prior_day="opposite"),
        "stop at the open": dict(stop="open"), "stop at first-hour middle": dict(stop="mid"),
        "stop at first-hour extreme": dict(stop="first-hour extreme"),
        "exit 14:00": dict(exit_hhmm=1400), "exit 15:00": dict(exit_hhmm=1500),
    }
    rows = []
    for mk in ("NQ", "RTY", "ES", "YM"):
        ds = [g.Days(a) for a in GROUPS[mk]]
        for vname, kw in variants.items():
            x = pd.concat([r9(d, COST[mk], **kw) for d in ds], ignore_index=True)
            st = stats(x, P3)
            rows.append(dict(market=mk, variant=vname, **st))
    res = pd.DataFrame(rows)
    res.to_csv(D + "r9_filters.csv", index=False)
    cols = [c for p, _, _ in P3 for c in (f"trades {p}", f"pf {p}", f"sh {p}")]
    print("\n=== B. R9 first-hour momentum filters ===")
    print(res[["market", "variant"] + [c for c in cols if c in res]].round(2).to_string(index=False))


if __name__ == "__main__":
    part_a()
    part_b()
