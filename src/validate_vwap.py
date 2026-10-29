"""Validate the two round-9 NQ VWAP candidates, then test them in the eval portfolio.
  A  sigma-band breakout, gap direction: first check (every E min, open+E .. close-60) with |close - VWAP| > k x
     volume-weighted sd in the gap's direction -> follow; exit back through VWAP at checks every X min, or hold to close
  B  weekly VWAP + gap direction: at open+m, price above (below) the Monday-anchored weekly VWAP [and today's VWAP] in
     the gap's direction -> follow; same exits
(1) parameter sweep: PF in 2010-14 (holdout) / 2015-22 / 2023-26, count positive in both later halves, train->test rank
    correlation of Sharpe (2015-22 -> 2023-26), cost stress for the round-9 originals.
(2) portfolio (2017-07 -> 2026-10, today's prices, like portfolio_sim2): noise+H19 2 MNQ + RSI fade 2 M2K, plus the
    ORIGINAL configs (not the sweep winners) at 1 MNQ: eval odds, LucidFlex program at 40% off, $2k Lucid ladder."""
import numpy as np
import pandas as pd

import capital_ladder_sim as CL
import firm_compare_sim as F
import lit_futures_grid as g
from holdout_eval import run_evals
from portfolio_sim2 import build, matrix
from research4 import GROUPS
from research9 import weekly_vwap
from config import DATA_DIR

D = DATA_DIR
pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 300)
PERIODS = (("2010-14", "2010-01-01", "2015-01-01"), ("2015-22", "2015-01-01", "2023-01-01"), ("2023-26", "2023-01-01", "2027-01-01"))


def prep(d):
    C, H, L, V, VW = d.C, d.H, d.L, d.V, d.vwap
    tp = (H + L + C) / 3
    cv, cpv, cp2v = np.cumsum(V, 1), np.cumsum(tp * V, 1), np.cumsum(tp * tp * V, 1)
    sd = np.sqrt(np.clip(np.where(cv > 0, cp2v / np.where(cv > 0, cv, 1), 0) - VW ** 2, 0, None))
    return dict(sd=sd, gap=np.sign(C[:, 0] - d.prev_close), wv=weekly_vwap(d, cpv, cv))


def exit_col(d, i, m, s, ex):
    """ex = 'hold' or vwap-cross checked every ex minutes (int)."""
    M = d.M
    if ex == "hold":
        return M
    a = np.arange(ex, M, ex)
    a = a[a > m]
    bad = s * (d.C[i, a] - d.vwap[i, a]) < 0
    return int(a[bad.argmax()]) if bad.any() else M


def out_row(name, d, i, m, s, x):
    p = d.C[i, m]
    return (name, d.dates[i], d.dates[i] + pd.Timedelta(minutes=d.open_min + x), s * (d.C[i, x] - p), p)


def band_trades(d, P, k, every, ex, name="band"):
    C, VW, M, sd, gap = d.C, d.vwap, d.M, P["sd"], P["gap"]
    a = np.arange(every, M - 59, every)
    out = []
    for i in np.where(~np.isnan(gap) & (gap != 0))[0]:
        z = gap[i] * (C[i, a] - VW[i, a]) / np.where(sd[i, a] > 0, sd[i, a], np.nan)
        hit = z > k
        if hit.any():
            m = int(a[hit.argmax()])
            s = int(gap[i])
            out.append(out_row(name, d, i, m, s, exit_col(d, i, m, s, ex)))
    return out


def weekly_trades(d, P, m, ex, daily=True, name="weekly"):
    C, VW, wv, gap = d.C, d.vwap, P["wv"], P["gap"]
    out = []
    for i in range(len(C)):
        if np.isnan(wv[i, m]) or np.isnan(gap[i]) or gap[i] == 0:
            continue
        s = int(gap[i])
        if s * (C[i, m] - wv[i, m]) > 0 and (not daily or s * (C[i, m] - VW[i, m]) > 0):
            out.append(out_row(name, d, i, m, s, exit_col(d, i, m, s, ex)))
    return out


def stats(tr, cost=1.0):
    bps = (tr.gross - cost) / tr.price * 1e4
    row = {}
    for p, lo, hi in PERIODS:
        x = bps[(tr.date >= lo) & (tr.date < hi)]
        day = x.groupby(tr.date[x.index]).sum()
        day = day.reindex(pd.bdate_range(lo, min(pd.Timestamp(hi), tr.date.max())), fill_value=0.0)
        row[f"pf_{p}"] = x[x > 0].sum() / -x[x < 0].sum()
        row[f"sh_{p}"] = day.mean() / day.std() * np.sqrt(252)
    row["trades_per_day"] = len(tr) / len(pd.bdate_range("2010-06-01", tr.date.max()))
    row["bps_trade"] = bps.mean()
    return row


if __name__ == "__main__":
    days = [g.Days(a) for a in GROUPS["NQ"]]
    preps = [prep(d) for d in days]
    rows = []
    for k in (0.5, 0.75, 1.0, 1.25, 1.5):
        for every in (15, 30, 60):
            for ex in (15, 30, 60, "hold"):
                tr = pd.DataFrame([r for d, P in zip(days, preps) for r in band_trades(d, P, k, every, ex)],
                                  columns=["strat", "date", "ts", "gross", "price"])
                rows.append(dict(cand="A band", config=f"k={k} checks {every}m exit {ex}", **stats(tr)))
    for m in (15, 30, 60, 90):
        for ex in (15, 30, 60, "hold"):
            for daily in (True, False):
                tr = pd.DataFrame([r for d, P in zip(days, preps) for r in weekly_trades(d, P, m, ex, daily)],
                                  columns=["strat", "date", "ts", "gross", "price"])
                rows.append(dict(cand="B weekly", config=f"open+{m} exit {ex}{'' if daily else ' no daily-VWAP check'}", **stats(tr)))
    sw = pd.DataFrame(rows)
    sw["both_halves"] = (sw["pf_2015-22"] > 1) & (sw["pf_2023-26"] > 1)
    sw["all3"] = sw.both_halves & (sw["pf_2010-14"] > 1)
    sw.to_csv(D + "validate_vwap_sweep.csv", index=False)
    print("=== (1) sweep ===")
    for c, s in sw.groupby("cand"):
        rc = s["sh_2015-22"].rank().corr(s["sh_2023-26"].rank())
        rh = s["sh_2015-22"].rank().corr(s["sh_2010-14"].rank())
        print(f"\n{c}: {len(s)} configs, positive both halves {s.both_halves.sum()}, all 3 periods {s.all3.sum()}, "
              f"rank corr 2015-22 -> 2023-26 {rc:.2f}, 2015-22 -> holdout {rh:.2f}, median PF "
              f"{s['pf_2010-14'].median():.2f} / {s['pf_2015-22'].median():.2f} / {s['pf_2023-26'].median():.2f}")
        print(s.drop(columns="cand").round(2).to_string(index=False))

    print("\n=== cost stress (round-9 originals) ===")
    orig = {"A band k=1 30m exit 30": lambda d, P: band_trades(d, P, 1.0, 30, 30),
            "B weekly open+30 exit 30": lambda d, P: weekly_trades(d, P, 30, 30)}
    for nm, fn in orig.items():
        tr = pd.DataFrame([r for d, P in zip(days, preps) for r in fn(d, P)], columns=["strat", "date", "ts", "gross", "price"])
        for ticks, cost in ((2, 0.5), (4, 1.0), (6, 1.5), (8, 2.0)):
            st = stats(tr, cost)
            print(f"{nm:28s} {ticks} ticks: PF {st['pf_2010-14']:.2f} / {st['pf_2015-22']:.2f} / {st['pf_2023-26']:.2f}")

    # (2) portfolio
    tr, dates = build()
    nq_now = g.Days("NQ").C[-1, -1]
    extra = []
    for d, P in zip(days[1:], preps[1:]):
        extra += band_trades(d, P, 1.0, 30, 30, "band") + weekly_trades(d, P, 30, 30, True, "weekly")
    ex = pd.DataFrame(extra, columns=["strat", "date", "ts", "gross", "price"])
    ex = ex[ex.date >= dates[0]]
    ex["usd"] = (ex.gross / ex.price * nq_now - 1.0) * 2.0
    tr = pd.concat([tr, ex], ignore_index=True)
    daily = tr.groupby(["date", "strat"])["usd"].sum().unstack(fill_value=0.0).reindex(dates, fill_value=0.0)
    print("\n=== (2) per 1 contract, 2017-07..2026-10 ===")
    st = pd.DataFrame({"trades_per_day": tr.groupby("strat").size() / len(dates), "usd_per_day": daily.mean(), "sd_day": daily.std()})
    st["sharpe"] = st.usd_per_day / st.sd_day * np.sqrt(252)
    print(st.round(2).to_string())
    print("\ndaily correlation:\n", daily.corr().round(2).to_string())
    ports = {"base: noise+H19 2 + RSI 2 M2K": dict(noise=2, rsi=2),
             "+ band 1 MNQ": dict(noise=2, rsi=2, band=1),
             "+ weekly 1 MNQ": dict(noise=2, rsi=2, weekly=1),
             "+ band 1 + weekly 1": dict(noise=2, rsi=2, band=1, weekly=1),
             "noise 1 + band 1 (swap) + RSI 2": dict(noise=1, rsi=2, band=1)}
    rng = np.random.default_rng(9)
    yrs = (dates[-1] - dates[0]).days / 365.25
    ra, rb, rc = [], [], []
    for pname, w in ports.items():
        m25 = matrix(tr, dates, w)
        pdaily = (daily[list(w)] * pd.Series(w)).sum(axis=1)
        for r in run_evals(m25, rng, "2017-2026"):
            if r["method"] == "historical starts":
                ra.append(dict(portfolio=pname, usd_day=pdaily.mean(), sd_day=pdaily.std(), **r))
        for plan, mult, keep in (("Lucid Flex 25K", 1.0, 0.0), ("Lucid Flex 50K", 1.5, 2000.0), ("Lucid Flex 150K", 3.5, 0.0)):
            p = dict(F.PLANS[plan], fee=F.PLANS[plan]["fee"] * 0.6, keep=keep)
            ev, fl = F.program(m25 * mult, dates, p)
            yr = fl.assign(y=dates[fl.di].year).groupby("y")["usd"].sum()
            rb.append(dict(portfolio=pname, plan=plan, pass_rate=(ev.res == "pass").mean(), net_per_yr=fl["usd"].sum() / yrs,
                           worst_year=yr.min(), losing_years=f"{(yr < 0).sum()}/{len(yr)}"))
        CL.TIERS = {"25K": ("Lucid Flex 25K", 1.0, 0.0), "50K": ("Lucid Flex 50K", 1.5, 2000.0), "150K": ("Lucid Flex 150K", 3.5, 0.0)}
        out = CL.Outcomes(m25)
        lad = CL.ladder([("50K", 3000), ("150K", 8000)])
        r = pd.DataFrame([CL.run(out, len(dates), s, 500, 2000, lad) for s in range(0, len(dates) - 500, 20)])
        rc.append(dict(portfolio=pname, lose_money=(r.end_cash < 2000).mean(), p10=r.end_cash.quantile(0.1), median=r.end_cash.median(),
                       p90=r.end_cash.quantile(0.9)))
    A = pd.DataFrame(ra)
    pct = [c for c in A.columns if c.startswith(("pass", "blow"))]
    print("\n=== A. eval odds ($1,250 / $1,000 DD, every real start day 2017-26) ===")
    print(A[["portfolio", "drawdown", "usd_day", "sd_day"] + pct + ["days_to_50pct"]].to_string(
        index=False, formatters={**{c: "{:.0%}".format for c in pct}, "usd_day": "${:.0f}".format, "sd_day": "${:.0f}".format}))
    print("\n=== B. LucidFlex weekly program, 40% off ===")
    print(pd.DataFrame(rb).to_string(index=False, formatters={"pass_rate": "{:.0%}".format, "net_per_yr": "${:,.0f}".format,
                                                             "worst_year": "${:,.0f}".format}))
    print("\n=== C. $2k Lucid ladder (2-year runs) ===")
    print(pd.DataFrame(rc).to_string(index=False, formatters={"lose_money": "{:.0%}".format, "p10": "${:,.0f}".format,
                                                             "median": "${:,.0f}".format, "p90": "${:,.0f}".format}))
    A.to_csv(D + "validate_vwap_odds.csv", index=False)
    pd.DataFrame(rb).to_csv(D + "validate_vwap_program.csv", index=False)
    pd.DataFrame(rc).to_csv(D + "validate_vwap_ladder.csv", index=False)
