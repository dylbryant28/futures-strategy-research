"""(1) TRUE HOLDOUT: the frozen portfolio on NQ 2010-06 -> 2014-12 (Databento, never used for any choice):
      noise  : noise area gapdir+tgt3+stop05          eia : Wednesday fade 10:30->10:45, hold to close
      wed    : Wednesday 10:00-10:30 direction -> last 30 min
      l30    : last-30 (prev close -> 15:30 signal, |p| > 0.5 sd), direction = trailing-60-day winner
    Per contract in points and bps, net of 1 pt, per year.
(2) REALISTIC EVAL: $1,250 target / $1,000 drawdown on MNQ, trades rescaled to today's NQ price, three ways:
      iid      : random days with replacement (what earlier sims used)
      block20  : random 20-day blocks (keeps streaks / regimes together)
      historical: an eval started on EVERY real trading day, run forward on the real sequence
    on 2015-2026 and on the 2010-2014 holdout."""
import numpy as np
import pandas as pd

import combo_eval_sim as ce
import lit_futures_grid as g
from noise_filters_nq import Ctx, simulate as noise_sim
from noise_walk_forward import kwargs_for
from config import DATA_DIR

g.ASSETS["NQ2010"] = dict(g.ASSETS["NQ"], files=["nq_2010_1min_et.parquet"])
g.ASSETS["NQ2015"] = dict(g.ASSETS["NQ"], files=["nq_2015_1min_et.parquet"])
STRATS = ["noise", "eia", "wed", "l30"]
MNQ_PV, COST, TARGET, MAX_DD = 2.0, 1.0, 1250.0, 1000.0
HORIZON, N_SIMS = 250, 20000
PORTFOLIOS = {"noise x2": dict(noise=2), "BASE (noise x2 + eia + wed)": dict(noise=2, eia=1, wed=1),
              "BASE + l30 x1": dict(noise=2, eia=1, wed=1, l30=1)}


def trades(assets):
    ds = [g.Days(a) for a in assets]
    out = []
    for d in ds:
        ctx = Ctx(d)
        t = noise_sim(ctx, **kwargs_for(["gapdir", "tgt3", "stop05"], ctx))
        out.append(pd.DataFrame(dict(strat="noise", gross=t["gross"], date=d.dates[t["day"].to_numpy()],
                                     price=d.C[t["day"].to_numpy(), 0],
                                     ts=d.dates[t["day"].to_numpy()] + pd.to_timedelta(d.open_min + t["xcol"], unit="m"))))
        lt = ce.lit_trades(d)
        out.append(lt[lt.strat.isin(["eia", "wed"])][["strat", "gross", "date", "price", "ts"]])
    parts = []
    for d in ds:  # last-30 with a trailing-60-day direction rule, run continuously across files
        M = d.M
        p = d.C[:, M - 30] / d.prev_close - 1
        mom = np.where(~np.isnan(p) & (p != 0), np.sign(p) * (d.C[:, M] - d.C[:, M - 30]), np.nan)
        parts.append(pd.DataFrame({"date": d.dates, "p": p, "mom": mom, "price": d.C[:, 0]}))
    x = pd.concat(parts, ignore_index=True)
    x["sig"] = x["p"].rolling(60, min_periods=20).std().shift(1)
    take = x["mom"].notna() & (x["p"].abs() > 0.5 * x["sig"])
    m = x["mom"].where(take, 0.0)
    dirn = np.sign(m.rolling(60, min_periods=60).sum().shift(1)).replace(0, np.nan)
    ok = take & dirn.notna()
    out.append(x[ok].assign(strat="l30", gross=(m * dirn)[ok], ts=x["date"][ok] + pd.Timedelta(hours=16))[
        ["strat", "gross", "date", "price", "ts"]])
    dates = pd.DatetimeIndex(np.concatenate([d.dates.values for d in ds])).unique().sort_values()
    return pd.concat(out, ignore_index=True), dates


def holdout_report(t, dates):
    rows = []
    for s in STRATS + ["portfolio BASE + l30 (noise x2)"]:
        if s.startswith("portfolio"):
            w = PORTFOLIOS["BASE + l30 x1"]
            x = t.assign(net=(t["gross"] - COST) * t["strat"].map(w))
        else:
            x = t[t.strat == s].assign(net=lambda z: z["gross"] - COST)
        day = x.groupby("date")["net"].sum().reindex(dates, fill_value=0.0)
        yr = day.groupby(day.index.year).sum()
        n = x["net"]
        rows.append(dict(strategy=s, trades=len(x), pf=n[n > 0].sum() / -n[n < 0].sum(),
                         sharpe=day.mean() / day.std() * np.sqrt(252), avg_pts=n.mean() if not s.startswith("portfolio") else np.nan,
                         avg_bps=((x["gross"] - COST) / x["price"] * 1e4).mean() if not s.startswith("portfolio") else np.nan,
                         total_pts=n.sum(), max_dd=(day.cumsum() - day.cumsum().cummax()).min(),
                         **{str(y): v for y, v in yr.round(0).items()}))
    return pd.DataFrame(rows)


def day_matrix(t, dates, px_now, weights):
    t = t[t["date"].isin(dates)].copy()
    t["usd"] = (t["gross"] / t["price"] * px_now - COST) * MNQ_PV * t["strat"].map(weights).fillna(0)
    t = t[t["usd"] != 0].sort_values(["date", "ts"], kind="stable")
    di = pd.Series(np.arange(len(dates)), index=dates)
    t["di"] = di[t["date"]].to_numpy()
    t["k"] = t.groupby("di").cumcount()
    mt = int(t["k"].max()) + 1
    m = np.zeros((len(dates), mt))
    m[t["di"], t["k"]] = t["usd"]
    return m


def outcome(paths, dd_type):
    """paths: (sims, days, trades) $ P&L. Returns pass_day, fail_day (inf if neither within the path)."""
    n, h, mt = paths.shape
    eq = np.cumsum(paths.reshape(n, -1), axis=1).reshape(n, h, mt)
    if dd_type == "static":
        thr = np.full((n, h), -MAX_DD)
    else:
        eod = eq[:, :, -1]
        peak = np.maximum.accumulate(np.concatenate([np.zeros((n, 1)), eod[:, :-1]], axis=1), axis=1)
        thr = np.minimum(np.maximum(peak, 0.0) - MAX_DD, 0.0)
    blow, win = (eq <= thr[:, :, None]).reshape(n, -1), (eq >= TARGET).reshape(n, -1)
    big = h * mt + 1
    bi = np.where(blow.any(1), blow.argmax(1), big)
    wi = np.where(win.any(1), win.argmax(1), big)
    return np.where(wi < bi, wi // mt + 1, np.inf), np.where(bi < wi, bi // mt + 1, np.inf)


def summarize(pd_, fd_, valid30=None):
    out = {}
    for d in (15, 30):
        mask = np.ones(len(pd_), bool) if valid30 is None else valid30[d]
        out[f"pass<={d}d"] = (pd_[mask] <= d).mean()
        out[f"blow<={d}d"] = (fd_[mask] <= d).mean()
    cum = np.array([(pd_ <= d).mean() for d in range(1, 121)])
    out["days_to_50pct"] = int(np.argmax(cum >= 0.5) + 1) if (cum >= 0.5).any() else np.nan
    out["n_evals"] = len(pd_)
    return out


def run_evals(m, rng, label):
    nd, mt = m.shape
    rows = []
    for dd in ("static", "trailing_eod"):
        # iid
        pick = rng.integers(0, nd, size=(N_SIMS, HORIZON))
        rows.append(dict(method="iid days", drawdown=dd, **summarize(*outcome(m[pick], dd))))
        # 20-day blocks (circular)
        nb = HORIZON // 20
        starts = rng.integers(0, nd, size=(N_SIMS, nb))
        pick = ((starts[:, :, None] + np.arange(20)[None, None, :]) % nd).reshape(N_SIMS, -1)
        rows.append(dict(method="20-day blocks", drawdown=dd, **summarize(*outcome(m[pick], dd))))
        # historical: every real start day, real order (paths truncated at the data end)
        idx = np.arange(nd)[:, None] + np.arange(HORIZON)[None, :]
        valid = idx < nd
        paths = np.where(valid[:, :, None], m[np.minimum(idx, nd - 1)], 0.0)
        pdays, fdays = outcome(paths, dd)
        remain = nd - np.arange(nd)
        rows.append(dict(method="historical starts", drawdown=dd,
                         **summarize(pdays, fdays, valid30={15: remain >= 15, 30: remain >= 30})))
    for r in rows:
        r["history"] = label
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    px_now = g.Days("NQ").C[-1, -1]
    t_hold, d_hold = trades(["NQ2010"])
    print(f"HOLDOUT NQ {d_hold[0].date()} -> {d_hold[-1].date()} ({len(d_hold)} days). Per contract, net of 1 pt:")
    print(holdout_report(t_hold, d_hold).round(2).to_string(index=False))

    t_main, d_main = trades(["NQ2015", "NQ"])
    rng = np.random.default_rng(5)
    rows = []
    for pname, w in PORTFOLIOS.items():
        for label, (tt, dd) in (("2015-2026", (t_main, d_main)), ("2010-2014 holdout", (t_hold, d_hold))):
            for r in run_evals(day_matrix(tt, dd, px_now, w), rng, label):
                rows.append(dict(portfolio=pname, **r))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "holdout_eval.csv", index=False)
    pct = [c for c in res.columns if c.startswith(("pass", "blow"))]
    cols = ["portfolio", "history", "drawdown", "method"] + pct + ["days_to_50pct", "n_evals"]
    print()
    print(res[cols].to_string(index=False, formatters={c: "{:.0%}".format for c in pct}))
