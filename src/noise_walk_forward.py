"""Walk-forward for the NQ noise-area candidates from noise_filters_nq.py: every on/off combination of six refinements,
split 70/30 at a weekend. All lookbacks (noise, vol percentile, noise percentile) are causal, so the run is done once on
the full data and trades are split by date. Per contract, net of 1 pt."""
import itertools

import numpy as np
import pandas as pd

from lit_futures_grid import Days
from noise_filters_nq import Ctx, simulate
from config import DATA_DIR

CANDIDATES = {
    "lowvol": "14d vol < 50th pct",
    "gapdir": "gap direction only",
    "nomon": "skip Mondays",
    "tgt3": "target 3x noise",
    "stop05": "stop 0.5x noise",
    "minnoise": "10:00 noise >= 33rd pct",
}


def kwargs_for(on, ctx):
    allow = np.ones(ctx.nd, dtype=bool)
    kw = {}
    if "lowvol" in on:
        allow &= ctx.rv_pct < 0.5
    if "nomon" in on:
        allow &= ctx.weekday != 0
    if "minnoise" in on:
        allow &= ctx.sig30_pct >= 0.33
    if "gapdir" in on:
        kw["gap_align"] = "with"
    if "tgt3" in on:
        kw["target_mult"] = 3.0
    if "stop05" in on:
        kw["stop_mult"] = 0.5
    kw["allow"] = allow
    return kw


def half_metrics(t, days_mask, cost, ctx):
    t = t[days_mask[t["day"].to_numpy()]]
    net = (t["gross"] - cost) * t["size"]
    day = np.zeros(ctx.nd)
    np.add.at(day, t["day"].to_numpy(), net.to_numpy())
    day = day[days_mask]
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    eq = np.cumsum(day)
    return dict(trades=len(t), pf=gains / losses if losses else np.nan,
                sharpe=day.mean() / day.std() * np.sqrt(252) if day.std() > 0 else np.nan,
                avg=net.mean() if len(t) else np.nan, total=net.sum(), dd=(eq - np.maximum.accumulate(eq)).min())


if __name__ == "__main__":
    d = Days("NQ")
    ctx = Ctx(d)
    dates = d.dates
    split = (dates.min() + (dates.max() - dates.min()) * 0.7 + pd.offsets.Week(weekday=5)).normalize()
    train, test = np.asarray(dates < split), np.asarray(dates >= split)
    print(f"NQ {dates[0].date()} -> {dates[-1].date()}; split {split.date()} "
          f"(train {train.sum()} days, test {test.sum()} days)")
    rows = []
    keys = list(CANDIDATES)
    for r in range(len(keys) + 1):
        for on in itertools.combinations(keys, r):
            t = simulate(ctx, **kwargs_for(on, ctx))
            row = dict(combo="+".join(on) or "base", n_filters=len(on))
            for name, mask in (("train", train), ("test", test), ("full", np.ones(ctx.nd, bool))):
                row.update({f"{k}_{name}": v for k, v in half_metrics(t, mask, d.cost, ctx).items()})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "noise_walk_forward.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    fmt = lambda x: f"{x:.2f}"
    show = ["combo", "trades_train", "pf_train", "sharpe_train", "avg_train", "trades_test", "pf_test", "sharpe_test",
            "avg_test", "total_test", "dd_test"]
    print("\nBase + each candidate alone:")
    print(res[res.n_filters <= 1][show].to_string(index=False, float_format=fmt))
    print("\nTop 10 by TRAIN Sharpe (>= 150 train trades) -> test:")
    print(res[res.trades_train >= 150].sort_values("sharpe_train", ascending=False).head(10)[show]
          .to_string(index=False, float_format=fmt))
    print("\nTop 5 by FULL-period Sharpe -> train vs test:")
    print(res.sort_values("sharpe_full", ascending=False).head(5)[show].to_string(index=False, float_format=fmt))
    both = ((res.pf_train > 1) & (res.pf_test > 1)).sum()
    beat = ((res.sharpe_test > res.loc[res.combo == "base", "sharpe_test"].iloc[0])).sum()
    rho = res[["sharpe_train", "sharpe_test"]].corr(method="spearman").iloc[0, 1]
    print(f"\n{both}/{len(res)} combos profitable in both halves; {beat - 0}/{len(res)} beat base Sharpe in test; "
          f"train-vs-test Sharpe rank corr {rho:.2f}")
