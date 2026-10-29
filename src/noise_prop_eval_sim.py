"""Monte Carlo of a prop-firm eval with the NQ noise-area strategy (gapdir+tgt3+stop05) on MNQ ($2/pt).
Rules: pass at +$1,250, fail at a $1,000 drawdown -- either STATIC (balance < start - 1000) or TRAILING EOD (limit
trails the highest end-of-day balance by $1,000 and locks at the starting balance). Breaches are checked on closed
trades (no intrabar unrealised P&L). Fixed 1, 2 or 3 MNQ per trade, 1 pt round-trip cost per contract.

Each historical trade is converted to a % move and rescaled to the latest NQ price, so 2015 trades (NQ ~4,000) count
at today's dollar size. Simulated evals draw whole trading days (including no-trade days) at random with replacement."""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from noise_compound_sim import trades_for
from config import DATA_DIR

g.ASSETS["NQ2015"] = dict(g.ASSETS["NQ"], files=["nq_2015_1min_et.parquet"])
COMBO = "gapdir+tgt3+stop05"
MNQ_PV, COST_PTS = 2.0, 1.0
TARGET, MAX_DD = 1250.0, 1000.0
N_SIMS, HORIZON = 20000, 250


def day_matrix(t, all_dates, px_now):
    """(n_days, max_trades) matrix of per-contract $ P&L per trade at today's price, 0-padded."""
    pts = t["gross"] / t["price"] * px_now - COST_PTS
    t = t.assign(usd=pts * MNQ_PV)
    per_day = t.groupby("date")["usd"].apply(list).reindex(all_dates)
    mx = int(per_day.dropna().map(len).max())
    m = np.zeros((len(all_dates), mx))
    for i, v in enumerate(per_day):
        if isinstance(v, list):
            m[i, :len(v)] = v
    return m


def simulate(m, contracts, dd_type, rng):
    nd, mt = m.shape
    pick = rng.integers(0, nd, size=(N_SIMS, HORIZON))
    pnl = m[pick] * contracts                                   # (sims, days, trades)
    eq = np.cumsum(pnl.reshape(N_SIMS, -1), axis=1).reshape(N_SIMS, HORIZON, mt)
    if dd_type == "static":
        thr = np.full((N_SIMS, HORIZON), -MAX_DD)
    else:
        eod = eq[:, :, -1]
        peak_before = np.maximum.accumulate(np.concatenate([np.zeros((N_SIMS, 1)), eod[:, :-1]], axis=1), axis=1)
        peak_before = np.maximum(peak_before, 0.0)
        thr = np.minimum(peak_before - MAX_DD, 0.0)
    blow = eq <= thr[:, :, None]
    win = eq >= TARGET
    big = HORIZON * mt + 1
    flat = lambda a: np.where(a.reshape(N_SIMS, -1).any(axis=1), a.reshape(N_SIMS, -1).argmax(axis=1), big)
    blow_i, win_i = flat(blow), flat(win)
    passed = win_i < blow_i
    failed = blow_i < win_i
    day_of = lambda i: i // mt + 1
    pass_day = np.where(passed, day_of(win_i), np.inf)
    fail_day = np.where(failed, day_of(blow_i), np.inf)
    return pass_day, fail_day


def summarize(pass_day, fail_day):
    out = {}
    for d in (15, 30):
        out[f"pass<={d}d"] = (pass_day <= d).mean()
        out[f"blow<={d}d"] = (fail_day <= d).mean()
        out[f"open@{d}d"] = 1 - out[f"pass<={d}d"] - out[f"blow<={d}d"]
    out["pass_eventually"] = np.isfinite(pass_day).mean()
    cum = np.array([(pass_day <= d).mean() for d in range(1, HORIZON + 1)])
    out["days_to_50pct_pass"] = int(np.argmax(cum >= 0.5) + 1) if (cum >= 0.5).any() else np.nan
    out["median_days_if_pass"] = np.median(pass_day[np.isfinite(pass_day)]) if np.isfinite(pass_day).any() else np.nan
    out["median_days_if_blow"] = np.median(fail_day[np.isfinite(fail_day)]) if np.isfinite(fail_day).any() else np.nan
    return out


if __name__ == "__main__":
    t = trades_for(COMBO)
    dates = pd.DatetimeIndex(np.concatenate([g.Days(a).dates.values for a in ("NQ2015", "NQ")])).unique().sort_values()
    nq_now = g.Days("NQ")
    px_now = nq_now.C[-1, -1]
    print(f"NQ now {px_now:,.2f}; {len(t)} trades over {len(dates)} trading days")
    histories = {"2015-2026": dates, "2021-2026": dates[dates >= "2021-01-01"]}
    rng = np.random.default_rng(7)
    rows = []
    for hname, hd in histories.items():
        m = day_matrix(t[t.date >= hd[0]], hd, px_now)
        per_ct = m.sum(axis=1)
        print(f"{hname}: per MNQ per day mean ${per_ct.mean():.2f}, sd ${per_ct.std():.2f}; "
              f"per trade median loss ${np.median(m[m < 0]):.0f}, median win ${np.median(m[m > 0]):.0f}")
        for dd_type in ("static", "trailing_eod"):
            for c in (1, 2, 3):
                rows.append(dict(history=hname, drawdown=dd_type, mnq=c, **summarize(*simulate(m, c, dd_type, rng))))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "noise_prop_eval_sim.csv", index=False)
    pd.set_option("display.width", 250)
    pct = [c for c in res.columns if c.startswith(("pass", "blow", "open"))]
    print(res.to_string(index=False, formatters={**{c: "{:.0%}".format for c in pct}}))
