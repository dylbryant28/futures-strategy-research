"""Research round 4: new MNQ strategy families built on different mechanics than the noise-area breakout, then the
same families on MGC, ES, CL, RTY, YM, ZN, 6E. All RTH, decisions on 5-min bar closes, exits on 1-min bars (stop
checked before target), net of 4 ticks, results in bps of notional. Periods 2010-14 (NQ only) / 2015-22 / 2023-26
(CL 2023-26 only); daily-P&L correlation with the NQ noise strategy and the average on its worst-10% days.
  F1 RSI2 pullback      : 5-min RSI(2) < 10 above a rising VWAP -> long (mirror short); exit RSI > 70/< 30 or time; stop 1x noise
  F2 exhaustion fade    : N consecutive same-direction 5-min bars covering >= 0.5x avg daily range -> fade 30 min
  F3 volume climax      : 5-min volume >= 3x its slot average and range >= 2x -> fade the bar (optionally only with a rejection wick)
  F4 VWAP pullback      : price held above (below) VWAP for >= 85% of the session so far, first touch of VWAP after the
                          first hour -> enter with the trend at VWAP, stop 0.5x noise beyond, exit at close
  F5 midday breakout    : break of the middle-third range in the last third of the session, stop at the other side, exit close
  F6 high-vol fade      : 14-day vol in its top third -> fade any 30-min move > 1 slot-sd, hold 30 min
  F7 day-after reversal : yesterday's |return| > 1.5 sd and the first 30 min move against it -> follow the reversal to the close
  F8 cross-asset lead   : (NQ target only) a > 1-sd 30-min move in ZN / 6E / MGC / ES -> trade NQ for the next 30 min,
                          sign fixed from 2015-22 data"""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from holdout_eval import trades as all_trades
from config import DATA_DIR

D = DATA_DIR
pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 500)

g.ASSETS.update({
    "NQ2010": dict(g.ASSETS["NQ"], files=["nq_2010_1min_et.parquet"]),
    "NQ2015": dict(g.ASSETS["NQ"], files=["nq_2015_1min_et.parquet"]),
    "ES2015": dict(g.ASSETS["ES"], files=["es_2015_1min_et.parquet"]),
    "RTY": dict(files=["rty_1min_et.parquet"], open=570, close=960, cost=0.4),
    "YM": dict(files=["ym_1min_et.parquet"], open=570, close=960, cost=4.0),
    "ZN": dict(files=["zn_1min_et.parquet"], open=500, close=900, cost=0.0625),
    "6E": dict(files=["6e_1min_et.parquet"], open=500, close=900, cost=0.0002),
})
GROUPS = {"NQ": ["NQ2010", "NQ2015", "NQ"], "MGC": ["MGC"], "ES": ["ES2015", "ES"], "CL": ["CL"], "RTY": ["RTY"],
          "YM": ["YM"], "ZN": ["ZN"], "6E": ["6E"]}


def exit_trade(d, i, m, s, p, stop=None, target=None, t_end=None, rsi_exit=None):
    """Walk 1-min bars after column m. stop/target are absolute prices (target may be an array per column)."""
    M = d.M
    te = M if t_end is None else min(t_end, M)
    if te <= m:
        return d.C[i, M]
    H, L, O = d.H[i, m + 1:te + 1], d.L[i, m + 1:te + 1], d.O[i, m + 1:te + 1]
    n = len(H)
    big = n + 5
    si = big
    if stop is not None:
        hit = (L <= stop) if s == 1 else (H >= stop)
        si = hit.argmax() if hit.any() else big
    ti = big
    if target is not None:
        tg = target[m + 1:te + 1] if isinstance(target, np.ndarray) else np.full(n, target)
        ok = ((H >= tg) if s == 1 else (L <= tg)) & ((tg - p) * s > 0)
        ti = ok.argmax() if ok.any() else big
    if rsi_exit is not None and rsi_exit < min(si, ti, n):
        return d.C[i, m + 1 + rsi_exit]
    if si == big and ti == big:
        return d.C[i, te]
    if si <= ti:
        return min(stop, O[si]) if s == 1 else max(stop, O[si])
    return max(tg[ti], O[ti]) if s == 1 else min(tg[ti], O[ti])


def prep(d):
    C, H, L, V, M = d.C, d.H, d.L, d.V, d.M
    nb = M // 5
    cols = np.arange(1, nb + 1) * 5
    c5 = C[:, cols]
    h5 = np.stack([H[:, c - 4:c + 1].max(1) for c in cols], 1)
    l5 = np.stack([L[:, c - 4:c + 1].min(1) for c in cols], 1)
    v5 = np.stack([V[:, c - 4:c + 1].sum(1) for c in cols], 1)
    o5 = np.concatenate([C[:, :1], c5[:, :-1]], 1)
    sig = pd.DataFrame(np.abs(C / C[:, :1] - 1)).rolling(14, min_periods=14).mean().shift(1).to_numpy()
    day_rng = H[:, 1:].max(1) - L[:, 1:].min(1)
    avg_rng = pd.Series(day_rng).rolling(14, min_periods=10).mean().shift(1).to_numpy()
    vavg = pd.DataFrame(v5).rolling(14, min_periods=10).mean().shift(1).to_numpy()
    ravg = pd.DataFrame(h5 - l5).rolling(14, min_periods=10).mean().shift(1).to_numpy()
    ret = C[:, -1] / d.prev_close - 1
    rv = pd.Series(ret).rolling(14, min_periods=10).std().shift(1)
    rvp = rv.rolling(253, min_periods=61).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).to_numpy()
    rsd = pd.Series(C[:, -1] / C[:, 0] - 1).rolling(60, min_periods=30).std().shift(1).to_numpy()
    return dict(cols=cols, c5=c5, h5=h5, l5=l5, v5=v5, o5=o5, sig=sig, avg_rng=avg_rng, vavg=vavg, ravg=ravg, rvp=rvp, rsd=rsd)


def rsi2(c5):
    d = np.diff(c5, axis=1, prepend=c5[:, :1])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    a = 0.5
    ru, rd = np.zeros_like(c5), np.zeros_like(c5)
    for b in range(1, c5.shape[1]):
        ru[:, b] = a * up[:, b] + (1 - a) * ru[:, b - 1]
        rd[:, b] = a * dn[:, b] + (1 - a) * rd[:, b - 1]
    return 100 - 100 / (1 + ru / np.where(rd == 0, 1e-12, rd))


def families(d, P, asset, cost):
    C, VW, M = d.C, d.vwap, d.M
    cols, c5, h5, l5, v5, o5 = P["cols"], P["c5"], P["h5"], P["l5"], P["v5"], P["o5"]
    nb = len(cols)
    out = []

    def rec(fam, var, i, m, s, px_exit, p):
        out.append((fam, var, d.dates[i], (s * (px_exit - p) - cost) / p * 1e4))

    noise_pts = P["sig"] * C[:, :1]
    first_bar = 6  # first hour of 5-min bars is warm-up for F1/F4
    # F1 RSI2 pullback
    R = rsi2(c5)
    vw5 = VW[:, cols]
    slope = vw5 - np.concatenate([vw5[:, :1].repeat(6, 1), vw5[:, :-6]], 1)
    for hold in (30, 60):
        for i in range(len(C)):
            if np.isnan(P["sig"][i, 0]):
                continue
            b, n = 3, 0
            while b < nb - 2 and n < 2:
                m = cols[b]
                s = 1 if (R[i, b] < 10 and c5[i, b] > vw5[i, b] and slope[i, b] > 0) else (
                    -1 if (R[i, b] > 90 and c5[i, b] < vw5[i, b] and slope[i, b] < 0) else 0)
                if s:
                    p = c5[i, b]
                    ex_rel = None
                    later = R[i, b + 1:]
                    hit = (later > 70) if s == 1 else (later < 30)
                    if hit.any():
                        ex_rel = (hit.argmax() + 1) * 5 - 1
                    px = exit_trade(d, i, m, s, p, stop=p - s * noise_pts[i, m], t_end=m + hold, rsi_exit=ex_rel)
                    rec("F1 RSI2 pullback", f"hold <= {hold} min", i, m, s, px, p)
                    n += 1
                    b += hold // 5
                b += 1
    # F2 exhaustion fade
    sgn5 = np.sign(c5 - o5)
    for N in (4, 5):
        for i in range(len(C)):
            if np.isnan(P["avg_rng"][i]):
                continue
            for b in range(N - 1, nb - 6):
                w = sgn5[i, b - N + 1:b + 1]
                if abs(w.sum()) == N and abs(c5[i, b] - o5[i, b - N + 1]) >= 0.5 * P["avg_rng"][i]:
                    s, p, m = -int(w[0]), c5[i, b], cols[b]
                    mv = abs(c5[i, b] - o5[i, b - N + 1])
                    px = exit_trade(d, i, m, s, p, stop=p - s * 0.5 * mv, t_end=m + 30)
                    rec("F2 exhaustion fade", f"{N} bars >= 0.5x avg range", i, m, s, px, p)
                    break
    # F3 volume climax
    for wick in (False, True):
        for hold in (30, 60):
            for i in range(len(C)):
                for b in range(1, nb - 6):
                    va, ra = P["vavg"][i, b], P["ravg"][i, b]
                    if not (va > 0 and ra > 0):
                        continue
                    rng = h5[i, b] - l5[i, b]
                    if v5[i, b] < 3 * va or rng < 2 * ra:
                        continue
                    up = c5[i, b] > o5[i, b]
                    pos = (c5[i, b] - l5[i, b]) / rng if rng > 0 else 0.5
                    if wick and not ((up and pos < 0.5) or (not up and pos > 0.5)):
                        continue
                    s, p, m = (-1 if up else 1), c5[i, b], cols[b]
                    stop = h5[i, b] if s == -1 else l5[i, b]
                    if (stop - p) * s >= 0:
                        continue
                    px = exit_trade(d, i, m, s, p, stop=stop, t_end=m + hold)
                    rec("F3 volume climax", f"{'rejection wick' if wick else 'any'} hold {hold}", i, m, s, px, p)
                    break
    # F4 VWAP pullback in trend
    above = (C[:, 1:] > VW[:, 1:]).cumsum(1) / np.arange(1, M + 1)
    for i in range(len(C)):
        if np.isnan(P["sig"][i, 0]):
            continue
        for m in range(60, M - 30):
            fr = above[i, m - 1]
            if fr >= 0.85 and d.L[i, m] <= VW[i, m] < d.C[i, m - 1]:
                s = 1
            elif fr <= 0.15 and d.H[i, m] >= VW[i, m] > d.C[i, m - 1]:
                s = -1
            else:
                continue
            p = VW[i, m]
            px = exit_trade(d, i, m, s, p, stop=p - s * 0.5 * noise_pts[i, m])
            rec("F4 VWAP pullback in trend", "first touch after 1st hour", i, m, s, px, p)
            break
    # F5 midday range breakout
    a, b_ = M // 3, 2 * M // 3
    hi_m, lo_m = d.H[:, a:b_].max(1), d.L[:, a:b_].min(1)
    for i in range(len(C)):
        for m in range(b_, M - 5):
            up, dn = d.H[i, m] >= hi_m[i], d.L[i, m] <= lo_m[i]
            if up == dn:
                if up:
                    break
                continue
            s = 1 if up else -1
            p = max(hi_m[i], d.O[i, m]) if up else min(lo_m[i], d.O[i, m])
            px = exit_trade(d, i, m, s, p, stop=lo_m[i] if up else hi_m[i])
            rec("F5 midday range breakout", "middle-third range, stop other side", i, m, s, px, p)
            break
    # F6 high-vol fade of 30-min moves
    r30 = c5[:, 6:] / c5[:, :-6] - 1
    sd30 = pd.DataFrame(r30).rolling(60, min_periods=30).std().shift(1).to_numpy()
    for i in np.where(P["rvp"] >= 0.67)[0]:
        for b in range(0, r30.shape[1] - 6, 6):
            if np.isnan(sd30[i, b]) or abs(r30[i, b]) <= sd30[i, b]:
                continue
            bb = b + 6
            s, p, m = -int(np.sign(r30[i, b])), c5[i, bb], cols[bb]
            px = exit_trade(d, i, m, s, p, t_end=m + 30)
            rec("F6 high-vol 30-min fade", "vol top third, |30m move| > 1 sd", i, m, s, px, p)
    # F7 day-after reversal
    yret = np.r_[np.nan, (C[:, -1] / C[:, 0] - 1)[:-1]]
    for i in range(1, len(C)):
        if np.isnan(P["rsd"][i]) or abs(yret[i]) <= 1.5 * P["rsd"][i]:
            continue
        s = -int(np.sign(yret[i]))
        if np.sign(C[i, 30] - C[i, 0]) != s:
            continue
        p = C[i, 30]
        stop = d.L[i, 1:31].min() if s == 1 else d.H[i, 1:31].max()
        px = exit_trade(d, i, 30, s, p, stop=stop)
        rec("F7 day-after reversal", "yesterday > 1.5 sd, first 30 min confirms", i, 30, s, px, p)
    return out


def cross_asset(nq_days, other_days, other, nq_cost=1.0):
    """NQ 30-min trades signalled by another market's preceding 30-min move."""
    idx = {dt: (d, i) for d in other_days for i, dt in enumerate(d.dates)}
    rec = []
    for d in nq_days:
        for i, dt in enumerate(d.dates):
            if dt not in idx:
                continue
            od, j = idx[dt]
            for hhmm in (1000, 1030, 1100, 1130, 1200, 1230, 1300, 1330, 1400, 1430):
                oc, oc0 = od.col(hhmm), od.col(hhmm - 30 if hhmm % 100 == 30 else hhmm - 70)
                if oc > od.M or oc0 < 0:
                    continue
                r_o = od.C[j, oc] / od.C[j, oc0] - 1
                m = d.col(hhmm)
                if m + 30 > d.M:
                    continue
                rec.append((dt, hhmm, r_o, d.C[i, m + 30] - d.C[i, m], d.C[i, m]))
    x = pd.DataFrame(rec, columns=["date", "slot", "r_o", "fwd_pts", "px"])
    x["sd"] = x.groupby("slot")["r_o"].transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
    train = x[x.date < "2023-01-01"]
    sign = np.sign(np.corrcoef(train["r_o"], train["fwd_pts"] / train["px"])[0, 1])
    t = x[x["r_o"].abs() > x["sd"]]
    s = sign * np.sign(t["r_o"])
    bps = (s * t["fwd_pts"] - nq_cost) / t["px"] * 1e4
    return [("F8 cross-asset lead", f"{other} 30-min move > 1 sd -> NQ next 30 min (sign {'+' if sign > 0 else '-'})", dt, b)
            for dt, b in zip(t["date"], bps)]


def summarize(tr, noise, bad):
    rows = []
    for (fam, asset, var), s in tr.groupby(["family", "asset", "variant"], sort=False):
        day = s.groupby("date")["bps"].sum()
        dd_all = day.reindex(noise.index, fill_value=0.0)
        mask = (dd_all != 0) | (noise != 0)
        corr = np.corrcoef(dd_all[mask], noise[mask])[0, 1] if mask.sum() > 30 else np.nan
        row = dict(family=fam, asset=asset, variant=var, corr_noise=corr,
                   bps_on_noise_bad=s[s.date.isin(bad)]["bps"].mean())
        ok = True
        for p, lo, hi in (("2010-14", "2010-01-01", "2015-01-01"), ("2015-22", "2015-01-01", "2023-01-01"), ("2023-26", "2023-01-01", "2027-01-01")):
            x = s[(s.date >= lo) & (s.date < hi)]
            if len(x) < 40:
                continue
            dd = day[(day.index >= lo) & (day.index < hi)].reindex(pd.bdate_range(lo, min(pd.Timestamp(hi), s.date.max())), fill_value=0.0)
            pf = x.bps[x.bps > 0].sum() / -x.bps[x.bps < 0].sum()
            row.update({f"n_{p}": len(x), f"pf_{p}": pf, f"bps_{p}": x.bps.mean(), f"sh_{p}": dd.mean() / dd.std() * np.sqrt(252)})
            ok &= pf > 1
        row["positive_all_periods"] = ok and row.get("pf_2023-26", 0) > 1
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t, nd = all_trades(["NQ2010", "NQ2015", "NQ"])
    noise = t[t.strat == "noise"].assign(net=lambda z: z.gross - 1.0).groupby("date")["net"].sum().reindex(nd, fill_value=0.0)
    bad = noise[noise <= noise[noise != 0].quantile(0.10)].index
    days = {k: [g.Days(a) for a in v] for k, v in GROUPS.items()}
    recs = []
    for asset, ds in days.items():
        cost = g.ASSETS[GROUPS[asset][-1]]["cost"]
        for d in ds:
            P = prep(d)
            recs += [(f, asset, v, dt, b) for f, v, dt, b in families(d, P, asset, cost)]
        print(f"{asset} done", flush=True)
    for other in ("ZN", "6E", "MGC", "ES"):
        recs += [(f, "NQ", v, dt, b) for f, v, dt, b in cross_asset(days["NQ"], days[other], other)]
    tr = pd.DataFrame(recs, columns=["family", "asset", "variant", "date", "bps"])
    tr.to_parquet(D + "research4_trades.parquet", index=False)
    res = summarize(tr, noise, bad)
    res.to_csv(D + "research4.csv", index=False)
    cols = ["family", "asset", "variant", "n_2010-14", "pf_2010-14", "sh_2010-14", "n_2015-22", "pf_2015-22", "sh_2015-22",
            "n_2023-26", "pf_2023-26", "sh_2023-26", "corr_noise", "bps_on_noise_bad", "positive_all_periods"]
    print(res[[c for c in cols if c in res]].round(2).to_string(index=False))
