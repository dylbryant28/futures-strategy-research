"""$50k compounding sim of the top noise-area combos on NQ 2015-01 -> 2026-10 (Databento, two files run back to back).
Risk per trade = r% of current equity; contracts = risk$ / (stop distance pts x $20), fractional (MNQ = 0.1 NQ makes
this roughly achievable). Net of 1 pt per contract round trip. Losses can exceed r% when a stop fills through."""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from noise_filters_nq import Ctx, simulate
from noise_walk_forward import kwargs_for
from config import DATA_DIR

g.ASSETS["NQ2015"] = dict(g.ASSETS["NQ"], files=["nq_2015_1min_et.parquet"])
COMBOS = ["gapdir+tgt3+stop05", "gapdir+stop05", "tgt3+stop05", "nomon+tgt3+stop05", "nomon+stop05"]
PV, COST, START = 20.0, 1.0, 50_000.0


def trades_for(combo):
    out = []
    for asset in ("NQ2015", "NQ"):
        d = g.Days(asset)
        ctx = Ctx(d)
        t = simulate(ctx, **kwargs_for(combo.split("+"), ctx))
        t["date"] = d.dates[t["day"].to_numpy()]
        t["price"] = d.C[t["day"].to_numpy(), 0]
        t["rv14"] = ctx.rv14[t["day"].to_numpy()]
        t["rv_pct"] = ctx.rv_pct[t["day"].to_numpy()]
        out.append(t)
    return pd.concat(out, ignore_index=True).sort_values(["date", "col"], kind="stable")


def run(t, risk, start_date=None):
    if start_date is not None:
        t = t[t["date"] >= start_date]
    eq, peak, max_dd, rows, max_lev = START, START, 0.0, [], 0.0
    for r in t.itertuples():
        n = eq * risk / (r.risk * PV)
        max_lev = max(max_lev, n * r.price * PV / eq)
        eq += n * (r.gross - COST) * PV
        peak = max(peak, eq)
        max_dd = min(max_dd, eq / peak - 1)
        rows.append((r.date, eq))
    e = pd.Series([x[1] for x in rows], index=pd.DatetimeIndex([x[0] for x in rows]))
    years = (e.index[-1] - e.index[0]).days / 365.25
    yearly = e.groupby(e.index.year).last()
    prev = pd.concat([pd.Series([START]), yearly.iloc[:-1].reset_index(drop=True)])
    yr_ret = pd.Series((yearly.to_numpy() / prev.to_numpy() - 1) * 100, index=yearly.index)
    return dict(final=eq, pnl=eq - START, cagr=((eq / START) ** (1 / years) - 1) * 100, max_dd=max_dd * 100,
                max_notional_x=max_lev), yr_ret


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    summary, yearly = [], {}
    for combo in COMBOS:
        t = trades_for(combo)
        for risk in (0.01, 0.02):
            for label, sd in (("2015-2026", None), ("2023-2026", pd.Timestamp("2023-01-01"))):
                s, yr = run(t, risk, sd)
                summary.append(dict(combo=combo, risk=f"{risk:.0%}", period=label, trades=len(t if sd is None else t[t.date >= sd]), **s))
                if sd is None:
                    yearly[(combo, f"{risk:.0%}")] = yr
    res = pd.DataFrame(summary)
    res.to_csv(DATA_DIR + "noise_compound_sim.csv", index=False)
    print(res.to_string(index=False, formatters={"final": "${:,.0f}".format, "pnl": "${:,.0f}".format,
                                                 "cagr": "{:.1f}%".format, "max_dd": "{:.1f}%".format,
                                                 "max_notional_x": "{:.1f}x".format}))
    print("\nYearly return % (2015 start):")
    print(pd.DataFrame(yearly).round(1).to_string())
