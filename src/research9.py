"""Research round 9: VWAP families aimed at the noise strategy's trade count (~0.3-1 trade/day), all 8 markets, RTH,
decisions on 1-min closes at the listed times, stops/targets on 1-min bars (stop first), flat at the close, net of 4
ticks, bps of notional. Periods 2010-14 (NQ only) / 2015-22 / 2023-26; corr with the NQ noise strategy.
Not repeated (already dead): VWAP trend (Zarattini & Aziz 2023), VWAP fade by avg range, inside-band VWAP fade,
RSI2 above rising VWAP, VWAP pullback in trend (RTY thin).
  V1 prior-day VWAP  : at open+30 price above BOTH yesterday's VWAP and today's VWAP -> long (mirror short); hold or
                       exit at a 30-min check back through today's VWAP; variant: the open also on that side
  V2 sigma-band break: first 30-min check (open+30 .. close-60) beyond VWAP +/- k x volume-weighted sd -> follow,
                       exit at a check back through VWAP or the close; variant: gap-direction only
  V3 sigma-band fade : first 1-min close beyond VWAP +/- k sd (open+60 .. close-60) -> fade, target VWAP, stop 1 sd
                       further, close exit
  V4 VWAP reclaim    : >= 90% of the last N min below VWAP, previous 15-min check below, this check above -> long
                       (mirror); stop at the low of the window; exit close or back through VWAP at a check
  V5 overnight VWAP  : VWAP of the overnight session (18:00 -> open); at open+30 price above overnight AND RTH VWAP
                       and the open above overnight VWAP -> long (mirror); hold / VWAP-cross exit
  V6 weekly VWAP     : VWAP anchored at Monday's RTH open (RTH only, same contract); at open+30 above weekly AND daily
                       VWAP -> long (mirror); hold / VWAP-cross exit
  V7 VWAP hold       : every 1-min close from open+15 to open+W on one side of VWAP -> enter at open+W that way
  V8 late VWAP gap   : at close-60 price > k sd from VWAP -> fade to VWAP (or follow) into the close
  V9 noise + VWAP    : refined noise (gapdir+tgt3+stop05) with the paper's band/VWAP trailing exit (NQ only)"""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from holdout_eval import trades as all_trades
from noise_filters_nq import Ctx, simulate as noise_sim
from noise_walk_forward import kwargs_for
from research4 import GROUPS, exit_trade
from config import DATA_DIR

D = DATA_DIR
pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 500)
COST = {k: g.ASSETS[v[-1]]["cost"] for k, v in GROUPS.items()}
REC = []  # (test, market, variant, date, bps, n)


def overnight_vwap(d, asset):
    a = g.ASSETS[asset]
    df = pd.concat([pd.read_parquet(D + f) for f in a["files"]], ignore_index=True)
    df = df.drop_duplicates("datetime_et", keep="last")
    t = df["datetime_et"]
    mod = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    on = (mod >= 18 * 60) | (mod <= a["open"])
    x = df[on]
    sess = x["datetime_et"].dt.normalize() + pd.to_timedelta((mod[on] >= 18 * 60).astype(int), unit="D")
    tp = (x["high"] + x["low"] + x["close"]) / 3
    gg = pd.DataFrame({"s": sess.to_numpy(), "pv": (tp * x["volume"]).to_numpy(), "v": x["volume"].to_numpy(),
                       "inst": x["instrument_id"].to_numpy()}).groupby("s")
    agg = gg.agg(pv=("pv", "sum"), v=("v", "sum"), inst=("inst", "last"), ninst=("inst", "nunique"))
    agg = agg.reindex(d.dates)
    ok = (agg.v > 0) & (agg.ninst == 1) & (agg.inst.to_numpy() == d.inst)
    return np.where(ok, agg.pv / agg.v.where(agg.v > 0), np.nan)


def weekly_vwap(d, cpv, cv):
    nd = len(d.dates)
    wk = d.dates.to_period("W").astype(str).to_numpy()
    prev_pv, prev_v = np.zeros(nd), np.zeros(nd)
    for i in range(1, nd):
        if wk[i] == wk[i - 1] and d.inst[i] == d.inst[i - 1]:
            prev_pv[i] = prev_pv[i - 1] + cpv[i - 1, -1]
            prev_v[i] = prev_v[i - 1] + cv[i - 1, -1]
    tot_v = prev_v[:, None] + cv
    w = (prev_pv[:, None] + cpv) / np.where(tot_v > 0, tot_v, np.nan)
    w[prev_v == 0] = np.nan  # Monday (or first day after a roll): weekly = daily, skip
    return w


def rec(test, mk, var, d, i, s, p, px, cost):
    REC.append((test, mk, var, d.dates[i], (s * (px - p) - cost) / p * 1e4, 1))


def families(d, mk, asset):
    cost = COST[mk]
    C, H, L, VW, V, M = d.C, d.H, d.L, d.vwap, d.V, d.M
    nd = len(C)
    tp = (H + L + C) / 3
    cv, cpv, cp2v = np.cumsum(V, 1), np.cumsum(tp * V, 1), np.cumsum(tp * tp * V, 1)
    sd = np.sqrt(np.clip(np.where(cv > 0, cp2v / np.where(cv > 0, cv, 1), 0) - VW ** 2, 0, None))
    chk = np.arange(30, M - 29, 30)
    gap = np.sign(C[:, 0] - d.prev_close)
    pdv = np.where(np.isnan(d.prev_close), np.nan, np.r_[np.nan, VW[:-1, -1]])
    onv = overnight_vwap(d, asset)
    wv = weekly_vwap(d, cpv, cv)

    def vx_exit(i, m, s):
        """Column of the first 30-min check after m with the close back through today's VWAP, else M."""
        a = chk[chk > m]
        bad = s * (C[i, a] - VW[i, a]) < 0
        return int(a[bad.argmax()]) if bad.any() else M

    def hold_or_vx(test, var, i, m, s):
        p = C[i, m]
        rec(test, mk, var + ", hold to close", d, i, s, p, C[i, M], cost)
        rec(test, mk, var + ", exit VWAP cross", d, i, s, p, C[i, vx_exit(i, m, s)], cost)

    m30 = 30
    for i in range(nd):
        c, vw = C[i, m30], VW[i, m30]
        # V1 prior-day VWAP
        if not np.isnan(pdv[i]):
            s = 1 if (c > pdv[i] and c > vw) else (-1 if (c < pdv[i] and c < vw) else 0)
            if s:
                hold_or_vx("V1 prior-day VWAP", "above/below both at open+30", i, m30, s)
                if s * (C[i, 0] - pdv[i]) > 0:
                    hold_or_vx("V1 prior-day VWAP", "+ open on the same side", i, m30, s)
        # V5 overnight VWAP
        if not np.isnan(onv[i]):
            s = 1 if (c > onv[i] and c > vw) else (-1 if (c < onv[i] and c < vw) else 0)
            if s:
                hold_or_vx("V5 overnight VWAP", "above/below both at open+30", i, m30, s)
                if s * (C[i, 0] - onv[i]) > 0:
                    hold_or_vx("V5 overnight VWAP", "+ open on the same side", i, m30, s)
        # V6 weekly VWAP
        if not np.isnan(wv[i, m30]):
            s = 1 if (c > wv[i, m30] and c > vw) else (-1 if (c < wv[i, m30] and c < vw) else 0)
            if s:
                hold_or_vx("V6 weekly VWAP", "above/below both at open+30", i, m30, s)
                if not np.isnan(gap[i]) and gap[i] == s:
                    hold_or_vx("V6 weekly VWAP", "+ gap direction", i, m30, s)
        # V2 sigma-band breakout (30-min checks)
        for k in (1.0, 1.5, 2.0):
            a = chk[chk <= M - 60]
            z = (C[i, a] - VW[i, a]) / np.where(sd[i, a] > 0, sd[i, a], np.nan)
            hit = np.abs(z) > k
            if hit.any():
                m = int(a[hit.argmax()])
                s = int(np.sign(z[hit.argmax()]))
                hold_or_vx("V2 sigma-band breakout", f"k={k}", i, m, s)
                if not np.isnan(gap[i]) and gap[i] == s:
                    hold_or_vx("V2 sigma-band breakout", f"k={k} gap direction", i, m, s)
        # V3 sigma-band fade (1-min)
        for k in (2.0, 2.5, 3.0):
            seg = slice(60, M - 59)
            z = (C[i, seg] - VW[i, seg]) / np.where(sd[i, seg] > 0, sd[i, seg], np.nan)
            hit = np.abs(z) > k
            if hit.any():
                m = 60 + int(hit.argmax())
                s = -int(np.sign(z[hit.argmax()]))
                p = C[i, m]
                px = exit_trade(d, i, m, s, p, stop=p - s * sd[i, m], target=VW[i])
                rec("V3 sigma-band fade", mk, f"k={k}, target VWAP, stop 1 sd", d, i, s, p, px, cost)
        # V4 VWAP reclaim (15-min checks)
        for N in (30, 60):
            for cc in range(max(N, 45), M - 89, 15):
                below = (C[i, cc - N:cc] < VW[i, cc - N:cc]).mean()
                s = 0
                if below >= 0.9 and C[i, cc - 15] < VW[i, cc - 15] and C[i, cc] > VW[i, cc]:
                    s, stop = 1, L[i, cc - N:cc + 1].min()
                elif below <= 0.1 and C[i, cc - 15] > VW[i, cc - 15] and C[i, cc] < VW[i, cc]:
                    s, stop = -1, H[i, cc - N:cc + 1].max()
                if s:
                    p = C[i, cc]
                    rec("V4 VWAP reclaim", mk, f"N={N}, stop window extreme, close", d, i, s, p,
                        exit_trade(d, i, cc, s, p, stop=stop), cost)
                    rec("V4 VWAP reclaim", mk, f"N={N}, stop window extreme, VWAP-cross exit", d, i, s, p,
                        exit_trade(d, i, cc, s, p, stop=stop, t_end=vx_exit(i, cc, s)), cost)
                    break
        # V7 VWAP hold
        for W in (45, 60, 90):
            side = np.sign(C[i, 15:W + 1] - VW[i, 15:W + 1])
            if (side == side[0]).all() and side[0] != 0:
                hold_or_vx("V7 VWAP hold", f"open+15 .. open+{W} one side", i, W, int(side[0]))
        # V8 late VWAP gap
        m = M - 60
        if sd[i, m] > 0:
            z = (C[i, m] - VW[i, m]) / sd[i, m]
            for k in (1.0, 1.5, 2.0):
                if abs(z) > k:
                    s = -int(np.sign(z))
                    p = C[i, m]
                    rec("V8 late VWAP gap", mk, f"k={k} fade, target VWAP", d, i, s, p,
                        exit_trade(d, i, m, s, p, target=VW[i]), cost)
                    rec("V8 late VWAP gap", mk, f"k={k} follow to close", d, i, -s, p, C[i, M], cost)


def v9(d):
    ctx = Ctx(d)
    for name, ex in (("band exit (base)", "band"), ("band/VWAP trailing exit", "vwap")):
        kw = dict(kwargs_for(["gapdir", "tgt3", "stop05"], ctx), exit=ex)
        t = noise_sim(ctx, **kw)
        for r in t.itertuples():
            p = d.C[r.day, 0]
            REC.append(("V9 noise + VWAP exit", "NQ", name, d.dates[r.day], (r.gross - 1.0) / p * 1e4, 1))


if __name__ == "__main__":
    t, nd_ = all_trades(["NQ2010", "NQ2015", "NQ"])
    noise = t[t.strat == "noise"].assign(net=lambda z: z.gross - 1.0).groupby("date")["net"].sum().reindex(nd_, fill_value=0.0)
    span = {}
    for mk, assets in GROUPS.items():
        ds = []
        for a in assets:
            d = g.Days(a)
            families(d, mk, a)
            if mk == "NQ":
                v9(d)
            ds.append(d)
        span[mk] = (min(d.dates.min() for d in ds), max(d.dates.max() for d in ds))
        print(mk, "done", flush=True)
    tr = pd.DataFrame(REC, columns=["test", "market", "variant", "date", "bps", "n"])
    tr.to_parquet(D + "research9_trades.parquet", index=False)
    rows = []
    for (te, mk, var), s in tr.groupby(["test", "market", "variant"], sort=False):
        day = s.groupby("date")["bps"].sum()
        da = day.reindex(noise.index, fill_value=0.0)
        mask = (da != 0) | (noise != 0)
        nbd = len(pd.bdate_range(*span[mk]))
        row = dict(test=te, market=mk, variant=var, trades_per_day=s.n.sum() / nbd, bps_trade=s.bps.mean(),
                   corr_noise=np.corrcoef(da[mask], noise[mask])[0, 1] if mask.sum() > 30 else np.nan)
        ok, nper = True, 0
        for p, lo, hi in (("2010-14", "2010-01-01", "2015-01-01"), ("2015-22", "2015-01-01", "2023-01-01"), ("2023-26", "2023-01-01", "2027-01-01")):
            x = s[(s.date >= lo) & (s.date < hi)]
            if len(x) < 50:
                continue
            dd = day[(day.index >= lo) & (day.index < hi)]
            dd = dd.reindex(pd.bdate_range(dd.index.min(), dd.index.max()), fill_value=0.0)
            pf = x.bps[x.bps > 0].sum() / -x.bps[x.bps < 0].sum()
            row.update({f"pf_{p}": pf, f"sh_{p}": dd.mean() / dd.std() * np.sqrt(252)})
            ok &= pf > 1
            nper += 1
        row["pass"] = bool(ok and nper >= 2)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(D + "research9.csv", index=False)
    cols = ["test", "market", "variant", "trades_per_day", "bps_trade", "pf_2010-14", "pf_2015-22", "sh_2015-22", "pf_2023-26",
            "sh_2023-26", "corr_noise", "pass"]
    print(f"{len(res)} variants, {res['pass'].sum()} pass")
    print(res[[c for c in cols if c in res]].round(2).to_string(index=False))
