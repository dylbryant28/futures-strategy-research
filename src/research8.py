"""Research round 8: multi-hour intraday rules with ~0.5-2+ trades/day (flat at the close), from the technical-trading
literature, on all 8 markets, 15-minute bars, net of 4 ticks (per side cost/2 on every position change), bps.
Results are per DAY (P&L summed per day) plus trades/day; periods 2010-14 (NQ) / 2015-22 / 2023-26; corr with NQ noise.
  R1 moving-average crossover (Brock, Lakonishok & LeBaron 1992): long fast > slow, short fast < slow (4/16, 8/32 bars)
  R2 trading-range break (BLL 1992): stop-and-reverse on a break of the last N-bar high/low (N = 16, 32)
  R3 Donchian channel (Turtle rules; Szakmary, Shen & Sharma 2010): enter on a 20-bar break, exit on a 10-bar opposite break
  R4 Keltner channel: EMA20 +/- 2 ATR break, exit back through the EMA
  R5 Bollinger fade: close outside 20-bar +/- 2 sd -> fade to the middle band; stop 1 sd further; 2-hour time stop
  R6 RSI(14) fade: < 25 long / > 75 short, exit at RSI 50 or 2 hours
  R7 initial-balance breakout: first close beyond the first-hour range, stop back through its middle, exit close
  R8 prior-day high/low breakout: first close beyond yesterday's RTH high/low, stop back inside, exit close
  R9 first-hour momentum: open -> 10:30 (session +60 min) move > 0.5 sd -> hold to the close
  R10 support/resistance pivots (Osler 2000): fade the first touch of floor-pivot R1 / S1, target the pivot, stop half way
      to R2 / S2, exit close"""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from holdout_eval import trades as all_trades
from research4 import GROUPS
from config import DATA_DIR

D = DATA_DIR
pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 500)
COST = {k: g.ASSETS[v[-1]]["cost"] for k, v in GROUPS.items()}
REC = []  # (test, market, variant, date, day_bps, n_trades)


def bars15(d):
    M = d.M
    cols = np.arange(15, M + 1, 15)
    c = d.C[:, cols]
    h = np.stack([d.H[:, x - 14:x + 1].max(1) for x in cols], 1)
    lo = np.stack([d.L[:, x - 14:x + 1].min(1) for x in cols], 1)
    return cols, c, h, lo


def flat(a):
    return a.reshape(-1)


def account(test, mk, var, d, cols, c, P, cost):
    """P: desired position after each bar close (days x bars); forced flat after the last bar."""
    nb = c.shape[1]
    P = P.copy()
    P[:, -1] = 0
    open0 = d.C[:, :1]
    px = np.concatenate([open0, c], 1)          # price at bar boundaries (col 0 = open)
    pos = np.concatenate([np.zeros((len(c), 1)), P], 1)  # position held from boundary k to k+1
    pnl = (pos[:, :-1] * (px[:, 1:] - px[:, :-1])).sum(1)
    changes = np.abs(np.diff(np.concatenate([np.zeros((len(c), 1)), P], 1), axis=1))
    entries = ((np.abs(P) > 0) & (np.concatenate([np.zeros((len(c), 1)), P[:, :-1]], 1) != P)).sum(1)
    net = (pnl - changes.sum(1) * cost / 2) / open0[:, 0] * 1e4
    for i in range(len(c)):
        if entries[i] or changes[i].sum():
            REC.append((test, mk, var, d.dates[i], net[i], int(entries[i])))


def state_machine(enter_long, enter_short, exit_long, exit_short):
    nd, nb = enter_long.shape
    P = np.zeros((nd, nb))
    pos = np.zeros(nd)
    for b in range(nb):
        pos = np.where((pos == 1) & exit_long[:, b], 0, pos)
        pos = np.where((pos == -1) & exit_short[:, b], 0, pos)
        pos = np.where((pos == 0) & enter_long[:, b], 1, pos)
        pos = np.where((pos == 0) & enter_short[:, b], -1, pos)
        P[:, b] = pos
    return P


def rules(d, mk):
    cost = COST[mk]
    cols, c, h, lo = bars15(d)
    nd, nb = c.shape
    s = pd.Series(flat(c))
    hs, ls = pd.Series(flat(h)), pd.Series(flat(lo))
    rs = lambda x: x.to_numpy().reshape(nd, nb)
    # R1
    for f, sl in ((4, 16), (8, 32)):
        P = np.sign(rs(s.rolling(f).mean() - s.rolling(sl).mean()))
        account("R1 MA crossover", mk, f"{f}/{sl} 15-min bars", d, cols, c, np.nan_to_num(P), cost)
    # R2
    for N in (16, 32):
        hi_n, lo_n = rs(hs.rolling(N).max().shift(1)), rs(ls.rolling(N).min().shift(1))
        sig = np.where(c > hi_n, 1.0, np.where(c < lo_n, -1.0, np.nan))
        P = pd.DataFrame(sig).ffill(axis=1).fillna(0).to_numpy()
        account("R2 trading-range break", mk, f"stop-and-reverse on {N}-bar break", d, cols, c, P, cost)
    # R3 Donchian 20/10
    hi20, lo20 = rs(hs.rolling(20).max().shift(1)), rs(ls.rolling(20).min().shift(1))
    hi10, lo10 = rs(hs.rolling(10).max().shift(1)), rs(ls.rolling(10).min().shift(1))
    P = state_machine(c > hi20, c < lo20, c < lo10, c > hi10)
    account("R3 Donchian channel", mk, "20-bar entry / 10-bar exit", d, cols, c, P, cost)
    # R4 Keltner
    ema = s.ewm(span=20, adjust=False).mean()
    tr = pd.concat([hs - ls, (hs - s.shift()).abs(), (ls - s.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up, dn, e = rs(ema + 2 * atr), rs(ema - 2 * atr), rs(ema)
    P = state_machine(c > up, c < dn, c < e, c > e)
    account("R4 Keltner channel", mk, "EMA20 +/- 2 ATR break, exit at EMA", d, cols, c, P, cost)
    # R5 Bollinger fade with 8-bar time stop (implemented with a per-bar age counter)
    m20, sd20 = rs(s.rolling(20).mean()), rs(s.rolling(20).std())
    P = np.zeros((nd, nb))
    pos, age, stp = np.zeros(nd), np.zeros(nd), np.zeros(nd)
    for b in range(nb):
        age = np.where(pos != 0, age + 1, 0)
        ex = ((pos == 1) & ((c[:, b] >= m20[:, b]) | (c[:, b] <= stp))) | ((pos == -1) & ((c[:, b] <= m20[:, b]) | (c[:, b] >= stp))) | (age >= 8)
        pos = np.where(ex, 0, pos)
        el = (pos == 0) & (c[:, b] < m20[:, b] - 2 * sd20[:, b])
        es = (pos == 0) & (c[:, b] > m20[:, b] + 2 * sd20[:, b])
        stp = np.where(el, c[:, b] - sd20[:, b], np.where(es, c[:, b] + sd20[:, b], stp))
        pos = np.where(el, 1, np.where(es, -1, pos))
        age = np.where(el | es, 0, age)
        P[:, b] = pos
    account("R5 Bollinger fade", mk, "20-bar 2 sd, target mid, 1 sd stop, 2h time stop", d, cols, c, P, cost)
    # R6 RSI(14) fade
    dlt = s.diff()
    gain, loss = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = rs(100 - 100 / (1 + gain / loss.replace(0, np.nan)))
    P = np.zeros((nd, nb))
    pos, age = np.zeros(nd), np.zeros(nd)
    for b in range(nb):
        age = np.where(pos != 0, age + 1, 0)
        ex = ((pos == 1) & (rsi[:, b] >= 50)) | ((pos == -1) & (rsi[:, b] <= 50)) | (age >= 8)
        pos = np.where(ex, 0, pos)
        el, es = (pos == 0) & (rsi[:, b] < 25), (pos == 0) & (rsi[:, b] > 75)
        pos = np.where(el, 1, np.where(es, -1, pos))
        age = np.where(el | es, 0, age)
        P[:, b] = pos
    account("R6 RSI(14) fade", mk, "< 25 / > 75, exit at 50 or 2h", d, cols, c, P, cost)
    # R7 initial balance breakout (first 4 bars = 60 min)
    ibh, ibl = h[:, :4].max(1)[:, None], lo[:, :4].min(1)[:, None]
    mid = (ibh + ibl) / 2
    after = np.arange(nb)[None, :] >= 4
    P = state_machine((c > ibh) & after, (c < ibl) & after, c < mid, c > mid)
    P = one_trade(P)
    account("R7 initial-balance breakout", mk, "first-hour range, stop at its middle, exit close", d, cols, c, P, cost)
    # R8 prior-day high/low breakout
    dh, dl = d.H[:, 1:].max(1), d.L[:, 1:].min(1)
    ph, pl = np.r_[np.nan, dh[:-1]][:, None], np.r_[np.nan, dl[:-1]][:, None]
    P = one_trade(state_machine(c > ph, c < pl, c < ph, c > pl))
    account("R8 prior-day high/low breakout", mk, "first close beyond, stop back inside, exit close", d, cols, c, P, cost)
    # R9 first-hour momentum
    r1 = c[:, 3] / d.C[:, 0] - 1
    sd = pd.Series(r1).rolling(60, min_periods=30).std().shift(1).to_numpy()
    P = np.zeros((nd, nb))
    P[:, 3:] = (np.sign(r1) * (np.abs(r1) > 0.5 * sd))[:, None]
    account("R9 first-hour momentum", mk, "|first-hour move| > 0.5 sd -> hold to close", d, cols, c, P, cost)
    # R10 floor pivots R1/S1 fade (minute resolution, first touch)
    C, H, L, O, M = d.C, d.H, d.L, d.O, d.M
    pc = np.r_[np.nan, C[:-1, -1]]
    pv = (np.r_[np.nan, dh[:-1]] + np.r_[np.nan, dl[:-1]] + pc) / 3
    R1, S1 = 2 * pv - np.r_[np.nan, dl[:-1]], 2 * pv - np.r_[np.nan, dh[:-1]]
    R2, S2 = pv + (np.r_[np.nan, dh[:-1]] - np.r_[np.nan, dl[:-1]]), pv - (np.r_[np.nan, dh[:-1]] - np.r_[np.nan, dl[:-1]])
    for i in range(1, nd):
        if np.isnan(pv[i]) or not (S1[i] < C[i, 0] < R1[i]):
            continue
        tu, td = H[i, 1:] >= R1[i], L[i, 1:] <= S1[i]
        iu = tu.argmax() + 1 if tu.any() else M + 1
        idn = td.argmax() + 1 if td.any() else M + 1
        if min(iu, idn) > M - 5:
            continue
        if iu < idn:
            sgn, m, p, tgt, stop = -1, iu, R1[i], pv[i], R1[i] + (R2[i] - R1[i]) / 2
        else:
            sgn, m, p, tgt, stop = 1, idn, S1[i], pv[i], S1[i] - (S1[i] - S2[i]) / 2
        hs_, ls_, os_ = H[i, m + 1:], L[i, m + 1:], O[i, m + 1:]
        hit_s = (hs_ >= stop) if sgn == -1 else (ls_ <= stop)
        hit_t = (ls_ <= tgt) if sgn == -1 else (hs_ >= tgt)
        a_ = hit_s.argmax() if hit_s.any() else 10 ** 6
        b_ = hit_t.argmax() if hit_t.any() else 10 ** 6
        if a_ == b_ == 10 ** 6:
            px = C[i, M]
        elif a_ <= b_:
            px = max(stop, os_[a_]) if sgn == -1 else min(stop, os_[a_])
        else:
            px = tgt
        REC.append(("R10 support/resistance pivots", mk, "fade first touch of R1/S1, target pivot", d.dates[i],
                    (sgn * (px - p) - cost) / p * 1e4, 1))


def one_trade(P):
    """Keep only the first position episode of each day."""
    out = P.copy()
    for i in range(len(P)):
        nz = np.flatnonzero(P[i])
        if len(nz):
            first = nz[0]
            end = first + np.argmax(P[i, first:] == 0) if (P[i, first:] == 0).any() else P.shape[1]
            out[i, end:] = 0
    return out


if __name__ == "__main__":
    days = {k: [g.Days(a) for a in v] for k, v in GROUPS.items()}
    t, nd_ = all_trades(["NQ2010", "NQ2015", "NQ"])
    noise = t[t.strat == "noise"].assign(net=lambda z: z.gross - 1.0).groupby("date")["net"].sum().reindex(nd_, fill_value=0.0)
    span = {}
    for mk, ds in days.items():
        for d in ds:
            rules(d, mk)
        span[mk] = (min(d.dates.min() for d in ds), max(d.dates.max() for d in ds))
        print(mk, "done", flush=True)
    tr = pd.DataFrame(REC, columns=["test", "market", "variant", "date", "bps", "n"])
    tr.to_parquet(D + "research8_days.parquet", index=False)
    rows = []
    for (te, mk, var), s in tr.groupby(["test", "market", "variant"], sort=False):
        day = s.groupby("date")["bps"].sum()
        da = day.reindex(noise.index, fill_value=0.0)
        mask = (da != 0) | (noise != 0)
        nbd = len(pd.bdate_range(*span[mk]))
        row = dict(test=te, market=mk, variant=var, trades_per_day=s.n.sum() / nbd,
                   corr_noise=np.corrcoef(da[mask], noise[mask])[0, 1] if mask.sum() > 30 else np.nan)
        ok, nper = True, 0
        for p, lo, hi in (("2010-14", "2010-01-01", "2015-01-01"), ("2015-22", "2015-01-01", "2023-01-01"), ("2023-26", "2023-01-01", "2027-01-01")):
            x = s[(s.date >= lo) & (s.date < hi)]
            if len(x) < 50:
                continue
            dd = day[(day.index >= lo) & (day.index < hi)]
            dd = dd.reindex(pd.bdate_range(dd.index.min(), dd.index.max()), fill_value=0.0)
            pf = x.bps[x.bps > 0].sum() / -x.bps[x.bps < 0].sum()
            row.update({f"pf_{p}": pf, f"sh_{p}": dd.mean() / dd.std() * np.sqrt(252), f"bps_day_{p}": dd.mean()})
            ok &= pf > 1
            nper += 1
        row["pass"] = bool(ok and nper >= 2)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(D + "research8.csv", index=False)
    cols = ["test", "market", "variant", "trades_per_day", "pf_2010-14", "pf_2015-22", "sh_2015-22", "pf_2023-26", "sh_2023-26", "corr_noise", "pass"]
    print(f"{len(res)} variants, {res['pass'].sum()} pass")
    print(res[[c for c in cols if c in res]].round(2).to_string(index=False))
