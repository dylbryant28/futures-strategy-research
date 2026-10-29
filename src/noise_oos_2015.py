"""Out-of-sample check of the top noise-area combos (chosen on 2022-12..2026-10) on NQ 2015-01..2022-12 (Databento).
Per contract, net of 1 pt round trip."""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from noise_filters_nq import Ctx, simulate
from noise_walk_forward import kwargs_for
from config import DATA_DIR

g.ASSETS["NQ2015"] = dict(g.ASSETS["NQ"], files=["nq_2015_1min_et.parquet"])
COMBOS = ["base", "stop05", "nomon+tgt3+stop05", "gapdir+tgt3+stop05", "tgt3+stop05", "gapdir+stop05", "nomon+stop05"]

if __name__ == "__main__":
    d = g.Days("NQ2015")
    ctx = Ctx(d)
    print(f"NQ {d.dates[0].date()} -> {d.dates[-1].date()}, {len(d.dates)} days")
    rows, yearly = [], {}
    for combo in COMBOS:
        on = [] if combo == "base" else combo.split("+")
        t = simulate(ctx, **kwargs_for(on, ctx))
        net = (t["gross"] - d.cost) * t["size"]
        day = np.zeros(ctx.nd)
        np.add.at(day, t["day"].to_numpy(), net.to_numpy())
        eq = np.cumsum(day)
        by_year = pd.Series(day).groupby(ctx.year).sum()
        yearly[combo] = by_year
        price = d.C[t["day"].to_numpy(), 0]
        rows.append(dict(combo=combo, trades=len(t), win=(net > 0).mean(),
                         pf=net[net > 0].sum() / -net[net < 0].sum(),
                         sharpe=day.mean() / day.std() * np.sqrt(252), avg_pts=net.mean(),
                         avg_bps=(net / price * 1e4).mean(), total_pts=net.sum(),
                         max_dd=(eq - np.maximum.accumulate(eq)).min(), yrs_pos=f"{(by_year > 0).sum()}/{len(by_year)}"))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "noise_oos_2015.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nNet points by year:")
    print(pd.DataFrame(yearly).round(0).to_string())
