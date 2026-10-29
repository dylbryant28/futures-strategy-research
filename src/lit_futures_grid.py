"""Simple versions of the six FUTURES_LITERATURE.md ideas, run as grids on NQ, ES, MGC and CL 1-min data.

Per contract, in points, net of a 4-tick round trip. All entries/exits at 1-min bar closes inside the regular session
(RTH) unless noted; flat at the RTH close except the month-end strategy. Day/trade validity:
  - full RTH sessions only (half days dropped), one contract (instrument_id) for the whole RTH session;
  - anything using the previous close or a multi-day hold is skipped when the contract rolled in between.

  S1 last30   : Gao et al. 2018 / Baltussen et al. 2021 -- predictor return -> trade the last w minutes in its direction
  S2 noise    : Zarattini, Aziz & Barbon 2024 -- break of the noise band around the open, band/VWAP trailing exit
  S3 orb      : Zarattini & Aziz 2023 -- first N-min range, 'dir' = enter in the range candle's direction at its close,
                'break' = stop entry at the range high/low; stop at the other end; target R x risk or EOD
  S4 openrev  : Grant, Wolf & Yu 2005 -- fade a large first-N-min move; gap fade = fade the overnight gap (Yu et al. 2005)
  S5 eia      : Geman & Li 2018 / Wen et al. 2023 -- Wednesdays 10:30 ET: fade the release reaction; mid-morning
                half-hour predicts the last half-hour
  S6 monthend : Etula et al. / Graziani 2024 -- month-end liquidity reversal
"""
import numpy as np
import pandas as pd
from config import DATA_DIR

ASSETS = {
    # file(s), RTH open / close (ET minutes of day), 4-tick round-trip cost in points
    "NQ": dict(files=["nq_db_1min_et.parquet"], open=9 * 60 + 30, close=16 * 60, cost=1.0),
    "ES": dict(files=["es_1min_et.parquet"], open=9 * 60 + 30, close=16 * 60, cost=1.0),
    "MGC": dict(files=["mgc_2015_1min_et.parquet", "mgc_1min_et.parquet"], open=8 * 60 + 20, close=13 * 60 + 30, cost=0.4),
    "CL": dict(files=["cl_1min_et.parquet"], open=9 * 60, close=14 * 60 + 30, cost=0.04),
}
DIR = DATA_DIR


# ----------------------------------------------------------------------------------------------------------------------
# Data: one row per RTH day, one column per minute from the open (col 0 = last price at/before the open) to the close.
# ----------------------------------------------------------------------------------------------------------------------
class Days:
    def __init__(self, asset):
        a = ASSETS[asset]
        df = pd.concat([pd.read_parquet(DIR + f) for f in a["files"]], ignore_index=True)
        df = df.drop_duplicates("datetime_et", keep="last").sort_values("datetime_et", kind="stable")
        t = df["datetime_et"]
        mod = (t.dt.hour * 60 + t.dt.minute).to_numpy()
        date = t.dt.normalize().to_numpy()
        o_min, c_min = a["open"], a["close"]
        ncol = c_min - o_min + 1

        rth = (mod > o_min) & (mod <= c_min) & (pd.DatetimeIndex(date).weekday < 5)
        r = df[rth]
        rdate, rcol = date[rth], mod[rth] - o_min
        g = pd.DataFrame({"date": rdate, "col": rcol, "inst": r["instrument_id"].to_numpy()})
        stats = g.groupby("date").agg(first=("col", "min"), last=("col", "max"), n=("col", "size"),
                                      inst0=("inst", "first"), inst1=("inst", "last"), ninst=("inst", "nunique"))
        ok = (stats["first"] <= 5) & (stats["last"] >= ncol - 3) & (stats["n"] >= 0.5 * ncol) & (stats["ninst"] == 1)
        stats = stats[ok]
        self.dates = pd.DatetimeIndex(stats.index)
        self.inst = stats["inst1"].to_numpy()
        nd = len(self.dates)
        di = pd.Series(np.arange(nd), index=self.dates)
        keep = np.isin(rdate, self.dates.values)
        rows = di[rdate[keep]].to_numpy()
        cols = rcol[keep]

        def grid(col):
            m = np.full((nd, ncol), np.nan)
            m[rows, cols] = r[col].to_numpy()[keep]
            return m

        C, H, L, O, V = grid("close"), grid("high"), grid("low"), grid("open"), grid("volume")
        # col 0: last trade at/before the open on the same calendar date (overnight session), else first RTH open
        pre = df[(mod <= o_min) & np.isin(date, self.dates.values)]
        pre_close = pre.groupby(pre["datetime_et"].dt.normalize())["close"].last()
        C[:, 0] = pre_close.reindex(self.dates).to_numpy()
        first_open = O[np.arange(nd), np.argmax(~np.isnan(O[:, 1:]), axis=1) + 1]
        C[:, 0] = np.where(np.isnan(C[:, 0]), first_open, C[:, 0])
        C = pd.DataFrame(C).ffill(axis=1).to_numpy()
        Cl = np.concatenate([C[:, :1], C[:, :-1]], axis=1)
        O = np.where(np.isnan(O), Cl, O)
        H = np.where(np.isnan(H), C, H)
        L = np.where(np.isnan(L), C, L)
        V = np.nan_to_num(V)
        tp = (H + L + C) / 3
        cv = np.cumsum(V, axis=1)
        self.vwap = np.where(cv > 0, np.cumsum(tp * V, axis=1) / np.where(cv > 0, cv, 1), C[:, :1])
        self.C, self.H, self.L, self.O, self.V = C, H, L, O, V
        self.M = ncol - 1
        self.open_min = o_min
        # previous RTH close, only when it is the same contract and at most 4 calendar days back
        gap_days = np.concatenate([[99], np.diff(self.dates.values).astype("timedelta64[D]").astype(int)])
        same = np.concatenate([[False], self.inst[1:] == self.inst[:-1]])
        self.prev_close = np.where(same & (gap_days <= 4), np.concatenate([[np.nan], C[:-1, -1]]), np.nan)
        self.cost = a["cost"]

    def col(self, hhmm):
        h, m = divmod(hhmm, 100)
        return h * 60 + m - self.open_min


def roll_sigma(x, n=60, minp=20):
    """Std of the previous n valid observations (no look-ahead)."""
    return pd.Series(x).rolling(n, min_periods=minp).std().shift(1).to_numpy()


def metrics(day_pnl_gross, n_trades_by_day, d, years):
    """day_pnl_gross: gross points booked per RTH day; costs = n trades x cost."""
    net = day_pnl_gross - n_trades_by_day * d.cost
    n = int(n_trades_by_day.sum())
    if n == 0:
        return dict(trades=0)
    tr = net[n_trades_by_day > 0]
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    eq = np.cumsum(net)
    by_year = pd.Series(net).groupby(years).sum()
    return dict(trades=n, win_days=(tr > 0).mean(), pf=gains / losses if losses else np.inf,
                gross_pts=day_pnl_gross.sum() / n, avg_pts=net.sum() / n, total_pts=net.sum(),
                sharpe=net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan,
                max_dd_pts=(eq - np.maximum.accumulate(eq)).min(), yrs_pos=f"{(by_year > 0).sum()}/{len(by_year)}")


# ----------------------------------------------------------------------------------------------------------------------
# Strategies: each returns (gross points per day, trades per day)
# ----------------------------------------------------------------------------------------------------------------------
def s1_last30(d, pred, k, w):
    C, M = d.C, d.M
    if pred == "prevclose_to_lastw":      # Baltussen: previous close -> start of the last window
        p = C[:, M - w] / d.prev_close - 1
    elif pred == "first30_incl_overnight":  # Gao: previous close -> open + 30 min
        p = C[:, 30] / d.prev_close - 1
    else:                                  # open -> start of the last window
        p = C[:, M - w] / C[:, 0] - 1
    sig = roll_sigma(p)
    take = ~np.isnan(p) & (p != 0) & ((k == 0) | (np.abs(p) > k * sig))
    pnl = np.where(take, np.sign(p) * (C[:, M] - C[:, M - w]), 0.0)
    return np.nan_to_num(pnl), take.astype(int)


def s2_noise(d, k, every, trail):
    C, M, V = d.C, d.M, d.vwap
    nd = len(C)
    A = np.abs(C / C[:, :1] - 1)
    sig = pd.DataFrame(A).rolling(14, min_periods=14).mean().shift(1).to_numpy()
    base_hi = np.fmax(C[:, 0], np.nan_to_num(d.prev_close, nan=-np.inf))
    base_lo = np.fmin(C[:, 0], np.nan_to_num(d.prev_close, nan=np.inf))
    U = base_hi[:, None] * (1 + k * sig)
    Lb = base_lo[:, None] * (1 - k * sig)
    pnl, ntr = np.zeros(nd), np.zeros(nd, dtype=int)
    checks = list(range(every, M, every))
    for i in range(nd):
        if np.isnan(sig[i, 0]):
            continue
        pos, entry = 0, 0.0
        for m in checks:
            p = C[i, m]
            if pos == 1 and p < (max(U[i, m], V[i, m]) if trail == "vwap" else U[i, m]):
                pnl[i] += p - entry; pos = 0
            elif pos == -1 and p > (min(Lb[i, m], V[i, m]) if trail == "vwap" else Lb[i, m]):
                pnl[i] += entry - p; pos = 0
            if pos == 0:
                if p > U[i, m]:
                    pos, entry = 1, p; ntr[i] += 1
                elif p < Lb[i, m]:
                    pos, entry = -1, p; ntr[i] += 1
        if pos:
            pnl[i] += pos * (C[i, M] - entry)
    return pnl, ntr


def _walk(d, i, start, s, entry, stop, target):
    """From minute `start`, exit at stop (worse of stop/open), target (better of target/open) or the close."""
    H, L, O, C, M = d.H[i], d.L[i], d.O[i], d.C[i], d.M
    for m in range(start, M + 1):
        if s == 1:
            if L[m] <= stop:
                return min(stop, O[m]) - entry
            if target is not None and H[m] >= target:
                return max(target, O[m]) - entry
        else:
            if H[m] >= stop:
                return entry - max(stop, O[m])
            if target is not None and L[m] <= target:
                return entry - min(target, O[m])
    return s * (C[M] - entry)


def s3_orb(d, n, mode, R):
    C, H, L, O, M = d.C, d.H, d.L, d.O, d.M
    nd = len(C)
    pnl, ntr = np.zeros(nd), np.zeros(nd, dtype=int)
    hi, lo = H[:, 1:n + 1].max(axis=1), L[:, 1:n + 1].min(axis=1)
    for i in range(nd):
        if mode == "dir":
            s = np.sign(C[i, n] - C[i, 0])
            if s == 0:
                continue
            entry, stop, start = C[i, n], (lo[i] if s == 1 else hi[i]), n + 1
        else:
            s = 0
            for m in range(n + 1, M + 1):
                up, dn = H[i, m] >= hi[i], L[i, m] <= lo[i]
                if up and dn:
                    break
                if up or dn:
                    s = 1 if up else -1
                    entry = max(hi[i], O[i, m]) if up else min(lo[i], O[i, m])
                    stop, start = (lo[i] if up else hi[i]), m + 1
                    break
            if s == 0:
                continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        target = None if R is None else entry + s * R * risk
        pnl[i] = _walk(d, i, start, s, entry, stop, target)
        ntr[i] = 1
    return pnl, ntr


def s4_openfade(d, n, k, hold):
    C, M = d.C, d.M
    r = C[:, n] / C[:, 0] - 1
    sig = roll_sigma(r)
    take = (r != 0) & (np.abs(r) > k * sig)
    exit_col = M if hold == "eod" else min(n + 60, M)
    pnl = np.where(take, -np.sign(r) * (C[:, exit_col] - C[:, n]), 0.0)
    return np.nan_to_num(pnl), take.astype(int)


def s4_gapfade(d, k, target):
    C, M = d.C, d.M
    g = C[:, 0] / d.prev_close - 1
    sig = roll_sigma(g)
    take = ~np.isnan(g) & (g != 0) & (np.abs(g) > k * sig)
    nd = len(C)
    pnl = np.zeros(nd)
    for i in np.where(take)[0]:
        s = -int(np.sign(g[i]))
        if target == "fill":
            # no stop: the 'stop' is placed out of reach so _walk only exits at the fill or the close
            pnl[i] = _walk(d, i, 1, s, C[i, 0], -np.inf if s == 1 else np.inf, d.prev_close[i])
        else:
            pnl[i] = s * (C[i, M] - C[i, 0])
    return pnl, take.astype(int)


def s5_eia_fade(d, x, k, hold):
    C, M = d.C, d.M
    t0 = d.col(1030)
    wed = d.dates.weekday == 2
    r = C[:, t0 + x] / C[:, t0] - 1
    sig = np.full(len(r), np.nan)
    sig[wed] = roll_sigma(r[wed], n=20, minp=10)
    take = wed & (r != 0) & ((k == 0) | (np.abs(r) > k * sig))
    exit_col = M if hold == "eod" else min(t0 + x + 30, M)
    pnl = np.where(take, -np.sign(r) * (C[:, exit_col] - C[:, t0 + x]), 0.0)
    return np.nan_to_num(pnl), take.astype(int)


def s5_eia_lasthalf(d, window, days):
    C, M = d.C, d.M
    a, b = (d.col(1000), d.col(1030)) if window == "10:00-10:30" else (d.col(1030), d.col(1100))
    p = C[:, b] / C[:, a] - 1
    wed = d.dates.weekday == 2
    sel = wed if days == "wed" else ~wed
    take = sel & (p != 0)
    pnl = np.where(take, np.sign(p) * (C[:, M] - C[:, M - 30]), 0.0)
    return pnl, take.astype(int)


def s6_monthend(d, variant, h):
    Cd, inst = d.C[:, -1], d.inst
    nd = len(Cd)
    per = d.dates.to_period("M")
    pnl, ntr = np.zeros(nd), np.zeros(nd, dtype=int)
    last_idx = pd.Series(np.arange(nd)).groupby(per).max().to_numpy()[:-1]  # drop the (possibly partial) last month
    for L in last_idx:
        if variant in ("rev_ls", "rev_long"):
            if L - 5 < 0 or L + h >= nd or inst[L - 5] != inst[L] or inst[L] != inst[L + h]:
                continue
            sig = Cd[L] / Cd[L - 5] - 1
            s = 1 if sig < 0 else (-1 if variant == "rev_ls" else 0)
            if s == 0:
                continue
            a, b = L, L + h
        elif variant == "short_T-5_to_T-1":
            a, b, s = L - 5, L - 1, -1
        else:  # long_T-1_to_T+3 (turn of month)
            a, b, s = L - 1, L + 3, 1
        if a < 0 or b >= nd or inst[a] != inst[b]:
            continue
        pnl[b] += s * (Cd[b] - Cd[a])
        ntr[b] += 1
    return pnl, ntr


GRIDS = {
    "S1_last30": (s1_last30, [dict(pred=p, k=k, w=w) for p in ("prevclose_to_lastw", "first30_incl_overnight", "open_to_lastw")
                              for k in (0, 0.5, 1.0) for w in (30, 60)]),
    "S2_noise": (s2_noise, [dict(k=k, every=e, trail=t) for k in (1.0, 1.5, 2.0) for e in (30, 15, 5)
                            for t in ("vwap", "band")]),
    "S3_orb": (s3_orb, [dict(n=n, mode=m, R=R) for n in (5, 15, 30) for m in ("dir", "break") for R in (None, 2, 5, 10)]),
    "S4_openfade": (s4_openfade, [dict(n=n, k=k, hold=h) for n in (5, 15, 30) for k in (1.0, 1.5, 2.0)
                                  for h in ("60m", "eod")]),
    "S4_gapfade": (s4_gapfade, [dict(k=k, target=t) for k in (0.5, 1.0, 1.5) for t in ("fill", "eod")]),
    "S5_eia_fade": (s5_eia_fade, [dict(x=x, k=k, hold=h) for x in (5, 15, 25) for k in (0, 1.0) for h in ("30m", "eod")]),
    "S5_eia_lasthalf": (s5_eia_lasthalf, [dict(window=w, days=dd) for w in ("10:00-10:30", "10:30-11:00")
                                          for dd in ("wed", "other")]),
    "S6_monthend": (s6_monthend, [dict(variant=v, h=h) for v in ("rev_ls", "rev_long") for h in (1, 3, 5, 10)]
                    + [dict(variant="short_T-5_to_T-1", h=0), dict(variant="long_T-1_to_T+3", h=0)]),
}


if __name__ == "__main__":
    import sys
    assets = sys.argv[1:] or list(ASSETS)
    rows = []
    for asset in assets:
        d = Days(asset)
        print(f"{asset}: {len(d.dates)} RTH days {d.dates[0].date()} -> {d.dates[-1].date()}", flush=True)
        years = d.dates.year.to_numpy()
        for strat, (fn, grid) in GRIDS.items():
            for params in grid:
                pnl, ntr = fn(d, **params)
                row = dict(strat=strat, asset=asset, config=" ".join(f"{k}={v}" for k, v in params.items()))
                row.update(metrics(pnl, ntr, d, years))
                rows.append(row)
    res = pd.DataFrame(rows)
    out = DIR + ("lit_futures_grid.csv" if len(assets) == len(ASSETS) else f"lit_futures_grid_{'_'.join(assets)}.csv")
    res.to_csv(out, index=False)
    print(f"wrote {out}")
