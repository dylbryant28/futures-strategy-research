"""Bankroll sim for buying LucidFlex evals with the NQ noise-area strategy, payouts reinvested.
Each run starts with $C cash on a real historical date and lasts 2 years (1 year on the 2010-14 holdout). Every 5 trading
days it buys one eval if: cash >= fee, < 10 evals open, < 5 funded open (Lucid household caps). Fees at 40% off.
Payouts (90% split already applied) land as cash on the payout day. Tiers (size, cushion) from earlier tests:
  25K: 2 MNQ, no cushion, $36 | 50K: 3 MNQ, $2k cushion, $88 | 100K: 4 MNQ, $2k cushion, $176 | 150K: 7 MNQ, $244
Policies: stay on one tier, or LADDER = buy the biggest tier whose 'upgrade bankroll' you have (cash >= threshold).
Same simplifications as firm_compare_sim (closed-trade checks, no live phase, rules from third-party summaries)."""
import numpy as np
import pandas as pd

import firm_compare_sim as F
import lit_futures_grid as g
from holdout_eval import trades, day_matrix
from config import DATA_DIR

TIERS = {"25K": ("Lucid Flex 25K", 2, 0.0), "50K": ("Lucid Flex 50K", 3, 2000.0),
         "100K": ("Lucid Flex 100K", 4, 2000.0), "150K": ("Lucid Flex 150K", 7, 0.0)}
DISC = 0.4


class Outcomes:
    """Cached eval / funded outcome for (tier, start day) on one history."""

    def __init__(self, base):
        self.base, self.cache = base, {}

    def get(self, tier, s):
        key = (tier, s)
        if key not in self.cache:
            pname, size, keep = TIERS[tier]
            p = dict(F.PLANS[pname], keep=keep)
            m = self.base * size
            res, end = F.run_eval(m, s, p)
            pays, fend = [], end
            if res == "pass":
                pays, fend, _ = F.run_funded(m, end + 1, p)
            self.cache[key] = (res, end, pays, fend)
        return self.cache[key]


def fee(tier):
    return F.PLANS[TIERS[tier][0]]["fee"] * (1 - DISC)


def run(out, nd, start, horizon, capital, policy):
    """policy(cash) -> tier to buy. Returns cash path stats."""
    cash, accounts, events = capital, [], []  # accounts: (s, eval_end, funded_end or None)
    stop = min(start + horizon, nd)
    payq = []  # (day, amount)
    min_cash, bought = cash, {t: 0 for t in TIERS}
    for t in range(start, stop):
        while payq and payq[0][0] <= t:
            cash += payq.pop(0)[1]
        if (t - start) % F.STEP:
            continue
        open_evals = sum(1 for a in accounts if a[0] <= t <= a[1])
        open_funded = sum(1 for a in accounts if a[2] is not None and a[1] < t <= a[2])
        tier = policy(cash)
        if tier is None or cash < fee(tier) or open_evals >= F.MAX_EVALS or open_funded >= 5:
            continue
        res, end, pays, fend = out.get(tier, t)
        cash -= fee(tier)
        bought[tier] += 1
        min_cash = min(min_cash, cash)
        funded_at_pass = sum(1 for a in accounts if a[2] is not None and a[1] < end <= a[2])
        if res == "pass" and funded_at_pass < 5:
            accounts.append((t, end, fend))
            payq = sorted(payq + [(d, x) for d, x in pays if d < stop])
        else:
            accounts.append((t, end, None))
    while payq and payq[0][0] < stop:
        cash += payq.pop(0)[1]
    return dict(end_cash=cash, min_cash=min_cash, **{f"n_{k}": v for k, v in bought.items()})


def ladder(thresholds):
    def pol(cash):
        tier = "25K"
        for t, th in thresholds:
            if cash >= th:
                tier = t
        return tier
    return pol


POLICIES = {
    "25K only": lambda c: "25K",
    "50K only": lambda c: "50K",
    "100K only": lambda c: "100K",
    "150K only": lambda c: "150K",
    "LADDER 25K -> 50K@$1.5k -> 100K@$3k -> 150K@$5k": ladder([("50K", 1500), ("100K", 3000), ("150K", 5000)]),
    "LADDER 25K -> 50K@$1k -> 100K@$2k -> 150K@$3k": ladder([("50K", 1000), ("100K", 2000), ("150K", 3000)]),
    "LADDER 25K -> 50K@$3k -> 150K@$8k": ladder([("50K", 3000), ("150K", 8000)]),
}

if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    px_now = g.Days("NQ").C[-1, -1]
    rows = []
    for hname, assets, horizon in (("2015-2026", ["NQ2015", "NQ"], 500), ("2010-2014", ["NQ2010"], 250)):
        t, dates = trades(assets)
        base = day_matrix(t[t.strat == "noise"], dates, px_now, dict(noise=1))
        out = Outcomes(base)
        starts = range(0, len(dates) - horizon, 20)
        for pname, pol in POLICIES.items():
            for cap in (250, 500, 1000, 2000, 5000):
                r = pd.DataFrame([run(out, len(dates), s, horizon, cap, pol) for s in starts])
                rows.append(dict(history=hname, policy=pname, capital=cap, runs=len(r),
                                 lose_money=(r.end_cash < cap).mean(), broke_at_some_point=(r.min_cash < fee("25K")).mean(),
                                 p10_end=r.end_cash.quantile(0.1), median_end=r.end_cash.median(), p90_end=r.end_cash.quantile(0.9),
                                 worst_end=r.end_cash.min(), lowest_cash=r.min_cash.min(),
                                 evals_bought=r[[c for c in r.columns if c.startswith("n_")]].sum(axis=1).mean()))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "capital_ladder_sim.csv", index=False)
    fm = {c: "${:,.0f}".format for c in ("p10_end", "median_end", "p90_end", "worst_end", "lowest_cash")}
    fm.update({"lose_money": "{:.0%}".format, "broke_at_some_point": "{:.0%}".format, "evals_bought": "{:.0f}".format})
    print(res.to_string(index=False, formatters=fm))
