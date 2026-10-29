"""Eval portfolio sims with the new sleeves (2017-07 -> 2026-10, the RTY data window), all at today's prices:
  noise  : NQ noise area gapdir+tgt3+stop05 + H19 (skip after |prior NQ day| > 2 sd), MNQ
  rsi    : RTY RSI(14) fade on 15-min bars (<25 / >75, exit at RSI 50 or 2 h), M2K ($5/pt)
  gapk1  : NQ first-hour momentum, gap direction, |first-hour move| > 1 sd, hold to close, MNQ
  first30: NQ first-30-min momentum, |move| > 0.5 sd, hold to close, MNQ
Trades are ordered by exit time inside each day; closed-trade drawdown checks; 1 pt (NQ) / 0.4 pt (RTY) per round trip.
Account sizing at 25K: noise 2 MNQ + listed extras; 50K = 1.5x, 150K = 3.5x (noise 3 / 7 MNQ).
  A. eval odds (every real start day, static / trailing EOD)  B. LucidFlex weekly program, 40% off  C. $2k Lucid ladder"""
import numpy as np
import pandas as pd

import capital_ladder_sim as CL
import firm_compare_sim as F
import lit_futures_grid as g
from holdout_eval import trades as noise_trades, run_evals
from research4 import GROUPS
from validate_r6_r9 import rsi_series
from config import DATA_DIR

D = DATA_DIR
START = pd.Timestamp("2017-07-10")
pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 300)


def h19_days(assets):
    c = pd.concat([pd.DataFrame({"date": d.dates, "r": d.C[:, -1] / d.prev_close - 1}) for d in (g.Days(a) for a in assets)])
    c = c.drop_duplicates("date").set_index("date")["r"]
    z = (c / c.rolling(60, min_periods=30).std()).shift(1)
    return set(z[z.abs() > 2].index)


def rsi_trades(d):
    cols = np.arange(15, d.M + 1, 15)
    c = d.C[:, cols]
    r = rsi_series(c, 14)
    out = []
    for i in range(len(c)):
        pos, age, ent, eb = 0, 0, 0.0, 0
        for b in range(len(cols)):
            if pos:
                age += 1
                last = b == len(cols) - 1
                if (pos == 1 and r[i, b] >= 50) or (pos == -1 and r[i, b] <= 50) or age >= 8 or last:
                    out.append(("rsi", d.dates[i], d.dates[i] + pd.Timedelta(minutes=d.open_min + int(cols[b])), pos * (c[i, b] - ent), ent))
                    pos = 0
                    continue
            if not pos and b < len(cols) - 1:
                if r[i, b] < 25:
                    pos, age, ent = 1, 0, c[i, b]
                elif r[i, b] > 75:
                    pos, age, ent = -1, 0, c[i, b]
    return out


def r9_trades(d, name, win, k, gap_dir):
    C, M = d.C, d.M
    r = C[:, win] / C[:, 0] - 1
    sd = pd.Series(r).rolling(60, min_periods=30).std().shift(1).to_numpy()
    gap = np.sign(C[:, 0] - d.prev_close)
    out = []
    for i in np.where(np.abs(r) > k * sd)[0]:
        s = np.sign(r[i])
        if gap_dir and gap[i] != s:
            continue
        out.append((name, d.dates[i], d.dates[i] + pd.Timedelta(minutes=d.open_min + M), s * (C[i, M] - C[i, win]), C[i, win]))
    return out


def build():
    nq_now, rty_now = g.Days("NQ").C[-1, -1], g.Days("RTY").C[-1, -1]
    t, dates = noise_trades(["NQ2015", "NQ"])
    skip = h19_days(["NQ2015", "NQ"])
    nz = t[(t.strat == "noise") & ~t.date.isin(skip)]
    rows = [("noise", r.date, r.ts, r.gross, r.price) for r in nz.itertuples()]
    for d in (g.Days(a) for a in GROUPS["NQ"][1:]):
        rows += r9_trades(d, "gapk1", 60, 1.0, True) + r9_trades(d, "first30", 30, 0.5, False)
    rows += rsi_trades(g.Days("RTY"))
    tr = pd.DataFrame(rows, columns=["strat", "date", "ts", "gross", "price"])
    nq = tr.strat != "rsi"
    tr["usd"] = np.where(nq, (tr.gross / tr.price * nq_now - 1.0) * 2.0, (tr.gross / tr.price * rty_now - 0.4) * 5.0)
    dates = dates[dates >= START]
    return tr[tr.date >= START], dates


def matrix(tr, dates, w):
    x = tr[tr.strat.isin(w)].copy()
    x["usd"] = x["usd"] * x["strat"].map(w)
    x = x.sort_values(["date", "ts"], kind="stable")
    di = pd.Series(np.arange(len(dates)), index=dates)
    x = x[x.date.isin(dates)]
    x["di"] = di[x["date"]].to_numpy()
    x["k"] = x.groupby("di").cumcount()
    m = np.zeros((len(dates), int(x.k.max()) + 1))
    m[x.di, x.k] = x.usd
    return m


PORTS = {
    "noise+H19 (base)": dict(noise=2),
    "+ RSI fade 2 M2K": dict(noise=2, rsi=2),
    "+ RSI fade 4 M2K": dict(noise=2, rsi=4),
    "+ gap-dir >1sd 1 MNQ": dict(noise=2, gapk1=1),
    "+ first-30 1 MNQ": dict(noise=2, first30=1),
    "+ RSI 2 M2K + gap-dir >1sd 1 MNQ": dict(noise=2, rsi=2, gapk1=1),
    "+ RSI 2 M2K + first-30 1 MNQ": dict(noise=2, rsi=2, first30=1),
    "+ RSI 4 M2K + first-30 1 MNQ": dict(noise=2, rsi=4, first30=1),
}

if __name__ == "__main__":
    tr, dates = build()
    daily = tr.groupby(["date", "strat"])["usd"].sum().unstack(fill_value=0.0).reindex(dates, fill_value=0.0)
    print("per 1 contract, 2017-07..2026-10 (today's prices):")
    st = pd.DataFrame({"trades_per_day": tr.groupby("strat").size() / len(dates), "usd_per_trade": tr.groupby("strat")["usd"].mean(),
                       "usd_per_day": daily.mean(), "sd_day": daily.std()})
    st["sharpe"] = st.usd_per_day / st.sd_day * np.sqrt(252)
    print(st.round(2).to_string())
    print("\ndaily correlation:\n", daily.corr().round(2).to_string())
    rng = np.random.default_rng(9)
    ra, rb, rc = [], [], []
    yrs = (dates[-1] - dates[0]).days / 365.25
    for pname, w in PORTS.items():
        m25 = matrix(tr, dates, w)
        pdaily = (daily[list(w)] * pd.Series(w)).sum(axis=1)
        for r in run_evals(m25, rng, "2017-2026"):
            if r["method"] == "historical starts":
                ra.append(dict(portfolio=pname, usd_day=pdaily.mean(), sd_day=pdaily.std(), **r))
        for plan, mult, keep in (("Lucid Flex 25K", 1.0, 0.0), ("Lucid Flex 50K", 1.5, 2000.0), ("Lucid Flex 150K", 3.5, 0.0)):
            p = dict(F.PLANS[plan], fee=F.PLANS[plan]["fee"] * 0.6, keep=keep)
            ev, fl = F.program(m25 * mult, dates, p)
            yr = fl.assign(y=dates[fl.di].year).groupby("y")["usd"].sum()
            rb.append(dict(portfolio=pname, plan=plan, pass_rate=(ev.res == "pass").mean(), net_per_eval=ev["paid"].mean() - p["fee"],
                           net_per_yr=fl["usd"].sum() / yrs, worst_year=yr.min(), losing_years=f"{(yr < 0).sum()}/{len(yr)}"))
        CL.TIERS = {"25K": ("Lucid Flex 25K", 1.0, 0.0), "50K": ("Lucid Flex 50K", 1.5, 2000.0), "150K": ("Lucid Flex 150K", 3.5, 0.0)}
        out = CL.Outcomes(m25)
        lad = CL.ladder([("50K", 3000), ("150K", 8000)])
        r = pd.DataFrame([CL.run(out, len(dates), s, 500, 2000, lad) for s in range(0, len(dates) - 500, 20)])
        rc.append(dict(portfolio=pname, lose_money=(r.end_cash < 2000).mean(), p10=r.end_cash.quantile(0.1), median=r.end_cash.median(),
                       p90=r.end_cash.quantile(0.9)))
    A = pd.DataFrame(ra)
    pct = [c for c in A.columns if c.startswith(("pass", "blow"))]
    print("\n=== A. eval odds ($1,250 target / $1,000 DD, every real start day 2017-26) ===")
    print(A[["portfolio", "drawdown", "usd_day", "sd_day"] + pct + ["days_to_50pct"]].to_string(
        index=False, formatters={**{c: "{:.0%}".format for c in pct}, "usd_day": "${:.0f}".format, "sd_day": "${:.0f}".format}))
    print("\n=== B. LucidFlex weekly program, 40% off ===")
    print(pd.DataFrame(rb).to_string(index=False, formatters={"pass_rate": "{:.0%}".format, "net_per_eval": "${:,.0f}".format,
                                                             "net_per_yr": "${:,.0f}".format, "worst_year": "${:,.0f}".format}))
    print("\n=== C. $2k Lucid ladder (2-year runs) ===")
    print(pd.DataFrame(rc).to_string(index=False, formatters={"lose_money": "{:.0%}".format, "p10": "${:,.0f}".format,
                                                             "median": "${:,.0f}".format, "p90": "${:,.0f}".format}))
    A.to_csv(D + "portfolio_sim2_odds.csv", index=False)
    pd.DataFrame(rb).to_csv(D + "portfolio_sim2_program.csv", index=False)
    pd.DataFrame(rc).to_csv(D + "portfolio_sim2_ladder.csv", index=False)
