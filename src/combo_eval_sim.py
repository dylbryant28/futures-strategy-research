"""Prop-eval Monte Carlo for a PORTFOLIO of the NQ strategies that were profitable, each on its own MNQ contract(s):
  noise   : noise area gapdir+tgt3+stop05 (noise_filters_nq)
  orb     : 15-min opening range, enter in the range candle's direction, stop at the other end, hold to close
  gapfade : fade the overnight gap when |gap| > 1.5 sd, hold to close
  eia     : Wednesdays, fade the 10:30->10:45 move, hold to close
  wed     : Wednesdays, 10:00-10:30 direction predicts the last 30 min
  fibvwap : FibVwap pick (original filters, 1-bar stop entry, close stop) -- fibvwap_nq_db_trades.parquet
Same eval rules and day-bootstrap as noise_prop_eval_sim ($1,250 target, $1,000 static or trailing-EOD drawdown,
closed-trade checks, trades rescaled to today's NQ price, 1 pt round trip per MNQ). Trades are ordered by exit time
within a day. Month-end (multi-day holds) is left out: prop evals generally require flat by the close."""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from noise_compound_sim import trades_for
from noise_prop_eval_sim import MNQ_PV, COST_PTS, TARGET, MAX_DD, HORIZON, summarize
from config import DATA_DIR

g.ASSETS["NQ2015"] = dict(g.ASSETS["NQ"], files=["nq_2015_1min_et.parquet"])
N_SIMS = 20000
STRATS = ["noise", "orb", "gapfade", "eia", "wed", "fibvwap", "breakout"]
BR_PICK = dict(atr_scale=1.0, entry_type="limit", stop_mode="hard")  # set from breakout_nq_runs results


def lit_trades(d):
    """Trades of the four lit-grid strategies with exit minute; gross points per contract."""
    C, H, L, O, M = d.C, d.H, d.L, d.O, d.M
    out = []
    # ORB 15 dir, hold to close
    n = 15
    hi, lo = H[:, 1:n + 1].max(axis=1), L[:, 1:n + 1].min(axis=1)
    for i in range(len(C)):
        s = np.sign(C[i, n] - C[i, 0])
        if s == 0:
            continue
        entry, stop = C[i, n], (lo[i] if s == 1 else hi[i])
        if entry == stop:
            continue
        px, xm = C[i, M], M
        for m in range(n + 1, M + 1):
            if s == 1 and L[i, m] <= stop:
                px, xm = min(stop, O[i, m]), m; break
            if s == -1 and H[i, m] >= stop:
                px, xm = max(stop, O[i, m]), m; break
        out.append(("orb", i, s * (px - entry), xm))
    # gap fade 1.5 sd, hold to close
    gap = C[:, 0] / d.prev_close - 1
    sig = g.roll_sigma(gap)
    for i in np.where(~np.isnan(gap) & (gap != 0) & (np.abs(gap) > 1.5 * sig))[0]:
        out.append(("gapfade", i, -np.sign(gap[i]) * (C[i, M] - C[i, 0]), M))
    wed = np.asarray(d.dates.weekday == 2)
    # EIA fade 10:30 -> 10:45, hold to close
    t0 = d.col(1030)
    r = C[:, t0 + 15] - C[:, t0]
    for i in np.where(wed & (r != 0))[0]:
        out.append(("eia", i, -np.sign(r[i]) * (C[i, M] - C[i, t0 + 15]), M))
    # Wednesday 10:00-10:30 direction -> last 30 min
    p = C[:, d.col(1030)] - C[:, d.col(1000)]
    for i in np.where(wed & (p != 0))[0]:
        out.append(("wed", i, np.sign(p[i]) * (C[i, M] - C[i, M - 30]), M + 0.5))
    t = pd.DataFrame(out, columns=["strat", "day", "gross", "xcol"])
    t["date"] = d.dates[t["day"].to_numpy()]
    t["price"] = d.C[t["day"].to_numpy(), 0]
    t["ts"] = t["date"] + pd.to_timedelta(d.open_min + t["xcol"], unit="m")
    return t


def all_trades():
    parts = []
    for asset in ("NQ2015", "NQ"):
        parts.append(lit_trades(g.Days(asset)))
    lit = pd.concat(parts, ignore_index=True)
    nz = trades_for("gapdir+tgt3+stop05")
    nz = pd.DataFrame(dict(strat="noise", gross=nz["gross"], date=nz["date"], price=nz["price"],
                           ts=nz["date"] + pd.to_timedelta(9 * 60 + 30 + nz["xcol"], unit="m")))
    fv = pd.read_parquet(DATA_DIR + "fibvwap_nq_db_trades.parquet")
    ext = pd.DatetimeIndex(fv["exit_time"])
    fv = pd.DataFrame(dict(strat="fibvwap", gross=fv["pts"], price=fv["entry"], ts=ext,
                           date=(ext + pd.Timedelta(hours=6)).normalize()))  # 18:00+ ET belongs to the next session
    br = pd.read_parquet(DATA_DIR + "breakout_nq_trades.parquet")
    for k, v in BR_PICK.items():
        br = br[br[k] == v]
    bx = pd.DatetimeIndex(br["exit_time"]) + pd.Timedelta(hours=3)  # engine runs on Pacific time
    br = pd.DataFrame(dict(strat="breakout", gross=br["pts"].to_numpy(), price=br["entry"].to_numpy(), ts=bx,
                           date=(bx + pd.Timedelta(hours=6)).normalize()))
    return pd.concat([nz, lit[nz.columns], fv, br], ignore_index=True)


def day_arrays(t, dates, px_now):
    t = t[t["date"].isin(dates)].copy()
    t["usd"] = (t["gross"] / t["price"] * px_now - COST_PTS) * MNQ_PV
    t["si"] = t["strat"].map({s: k for k, s in enumerate(STRATS)})
    t = t.sort_values(["date", "ts"], kind="stable")
    di = pd.Series(np.arange(len(dates)), index=dates)
    t["di"] = di[t["date"]].to_numpy()
    t["k"] = t.groupby("di").cumcount()
    mt = int(t["k"].max()) + 1
    val = np.zeros((len(dates), mt))
    strat = np.full((len(dates), mt), -1)
    val[t["di"], t["k"]] = t["usd"]
    strat[t["di"], t["k"]] = t["si"]
    return val, strat, t


def simulate(val, strat, weights, dd_type, rng):
    w = np.append(np.asarray(weights, float), 0.0)            # index -1 (padding) -> weight 0
    m = val * w[strat]
    nd, mt = m.shape
    pick = rng.integers(0, nd, size=(N_SIMS, HORIZON))
    eq = np.cumsum(m[pick].reshape(N_SIMS, -1), axis=1).reshape(N_SIMS, HORIZON, mt)
    if dd_type == "static":
        thr = np.full((N_SIMS, HORIZON), -MAX_DD)
    else:
        eod = eq[:, :, -1]
        peak = np.maximum.accumulate(np.concatenate([np.zeros((N_SIMS, 1)), eod[:, :-1]], axis=1), axis=1)
        thr = np.minimum(np.maximum(peak, 0.0) - MAX_DD, 0.0)
    blow, win = eq <= thr[:, :, None], eq >= TARGET
    big = HORIZON * mt + 1
    first = lambda a: np.where(a.reshape(N_SIMS, -1).any(1), a.reshape(N_SIMS, -1).argmax(1), big)
    bi, wi = first(blow), first(win)
    pass_day = np.where(wi < bi, wi // mt + 1, np.inf)
    fail_day = np.where(bi < wi, bi // mt + 1, np.inf)
    return pass_day, fail_day


PORTFOLIOS = {
    "noise x1": dict(noise=1),
    "noise x2": dict(noise=2),
    "noise+orb x1": dict(noise=1, orb=1),
    "noise+orb+gap+eia+wed x1": dict(noise=1, orb=1, gapfade=1, eia=1, wed=1),
    "all 6 x1": dict(noise=1, orb=1, gapfade=1, eia=1, wed=1, fibvwap=1),
    "all 6 x2": {s: 2 for s in STRATS},
    "noise+eia+wed x1": dict(noise=1, eia=1, wed=1),
    "noise x2 + eia+wed x1": dict(noise=2, eia=1, wed=1),
    "noise x2 + wed x1": dict(noise=2, wed=1),
    "noise x2 + eia+wed x1 + breakout x1": dict(noise=2, eia=1, wed=1, breakout=1),
}

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    t = all_trades()
    d_now = g.Days("NQ")
    px_now = d_now.C[-1, -1]
    dates = pd.DatetimeIndex(np.concatenate([g.Days("NQ2015").dates.values, d_now.dates.values])).unique().sort_values()
    val, strat, tt = day_arrays(t, dates, px_now)

    # standalone stats per strategy (per MNQ, today's price)
    daily = pd.DataFrame({s: pd.Series(tt[tt.strat == s].groupby("di")["usd"].sum()).reindex(range(len(dates)), fill_value=0.0)
                          for s in STRATS})
    st = pd.DataFrame({"trades": tt.groupby("strat").size(), "net_usd_per_trade": tt.groupby("strat")["usd"].mean(),
                       "pf": tt.groupby("strat")["usd"].apply(lambda x: x[x > 0].sum() / -x[x < 0].sum())}).reindex(STRATS)
    st["usd_per_day"] = daily.mean()
    st["sharpe"] = daily.mean() / daily.std() * np.sqrt(252)
    yrs = pd.Series(dates.year)
    st["yrs_pos"] = [f"{(daily[s].groupby(yrs).sum() > 0).sum()}/{yrs.nunique()}" for s in STRATS]
    print(f"NQ now {px_now:,.2f}; {len(dates)} days 2015-01 -> 2026-10. Per 1 MNQ, net:")
    print(st.round(2).to_string())
    print("\nDaily P&L correlation:")
    print(daily.corr().round(2).to_string())

    rng = np.random.default_rng(11)
    rows = []
    hist = {"2015-2026": np.ones(len(dates), bool), "2021-2026": np.asarray(dates >= "2021-01-01")}
    for hname, mask in hist.items():
        for pname, wts in PORTFOLIOS.items():
            weights = [wts.get(s, 0) for s in STRATS]
            pdaily = (daily * weights).sum(axis=1)[mask]
            for dd in ("static", "trailing_eod"):
                res = summarize(*simulate(val[mask], strat[mask], weights, dd, rng))
                rows.append(dict(history=hname, portfolio=pname, drawdown=dd, usd_day=pdaily.mean(),
                                 sd_day=pdaily.std(), **res))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "combo_eval_sim.csv", index=False)
    pct = [c for c in res.columns if c.startswith(("pass", "blow", "open"))]
    print()
    print(res.to_string(index=False, formatters={**{c: "{:.0%}".format for c in pct},
                                                 "usd_day": "${:.0f}".format, "sd_day": "${:.0f}".format}))
