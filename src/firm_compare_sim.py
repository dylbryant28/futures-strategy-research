"""Prop-firm comparison for the NQ noise-area strategy (gapdir+tgt3+stop05) on MNQ, weekly eval purchases on the REAL
day sequence (2015-2026 and the 2010-2014 holdout), trades rescaled to today's NQ price, 1 pt cost per MNQ round trip.
Rules from help-center articles / third-party summaries collected 2026-10-27 (they conflict in places; see notes).
Simplifications: drawdowns checked on closed trades (no unrealised P&L), 'intraday trailing' trails closed-trade
equity; soft daily loss limits stop trading for the day; no live-account phase; caps 10 open evals / 5 funded
(MFFU 100K+: 3 funded); an eval is bought every 5 trading days when there is room. Size = MNQ per $1k of max loss."""
import numpy as np
import pandas as pd

import lit_futures_grid as g
from holdout_eval import trades, day_matrix
from config import DATA_DIR

STEP, MAX_EVALS, FUNDED_DAYS = 5, 10, 750


def plan(**kw):
    base = dict(fee=None, target=None, dd=None, eval_lock=False, eval_cons=None, eval_dll=None, fdd=None, fdll=None,
                f_intraday=False, lock_trigger=None, split=0.9, max_payouts=None, max_funded=5, payout=None)
    base.update(kw)
    if base["fdd"] is None:
        base["fdd"] = base["dd"]
    if base["lock_trigger"] is None:
        base["lock_trigger"] = base["fdd"] + 100
    return base


PLANS = {}
# ---- LucidFlex: EOD trail, locks at +100 once EOD > +MLL+100 (eval and funded), 50% eval consistency, payout = 5 days
#      >= X + positive cycle, request min(50% profit, cap) >= $500, 90/10, 5 payouts then live. Fees: user $60 (25K), list.
for k, (fee, tgt, dd, day, cap) in {"25K": (60, 1250, 1000, 100, 1000), "50K": (146, 3000, 2000, 150, 2000),
                                     "100K": (293, 6000, 3000, 200, 2500), "150K": (407, 9000, 4500, 250, 3000)}.items():
    PLANS[f"Lucid Flex {k}"] = plan(fee=fee, target=tgt, dd=dd, eval_lock=True, eval_cons=0.5, max_payouts=5,
                                    payout=dict(kind="windays", days=5, day=day, pct=0.5, cap=cap, min=500))
# ---- Tradeify Growth: eval no consistency, soft DLL; funded DLL, 35% consistency, min balance, progressive caps.
for k, (fee, tgt, dd, dll, minbal, day, caps, mn) in {
        "25K": (99, 1500, 1000, 600, 1500, 100, [1000], 250), "50K": (145, 3000, 2000, 1250, 3000, 150, [1500, 2000, 2500, 3000], 500),
        "100K": (255, 6000, 3500, 2500, 4500, 200, [2000, 2500, 3000, 4000], 1000),
        "150K": (369, 9000, 5000, 3750, 6500, 250, [2500, 3000, 4000, 5000], 1500)}.items():
    PLANS[f"Tradeify Growth {k}"] = plan(fee=fee, target=tgt, dd=dd, eval_dll=dll, fdll=dll,
                                         payout=dict(kind="windays", days=5, day=day, pct=1.0, caps=caps, min=mn,
                                                     minbal=minbal, cons=0.35))
# ---- Tradeify Select Flex: eval 40% consistency, no DLL; funded 5 days >= X, 50% of profit, caps.
for k, (fee, tgt, dd, day, cap) in {"25K": (109, 1500, 1000, 100, 1250), "50K": (165, 3000, 2000, 150, 3000),
                                     "100K": (265, 6000, 3000, 200, 4000), "150K": (369, 9000, 4500, 250, 5000)}.items():
    PLANS[f"Tradeify Select Flex {k}"] = plan(fee=fee, target=tgt, dd=dd, eval_cons=0.4,
                                              payout=dict(kind="windays", days=5, day=day, pct=0.5, cap=cap, min=0))
# ---- Tradeify Select Daily: same eval; funded smaller MLL at 100K/150K, DLL, daily payout <= 2x fresh profit, cap, buffer.
for k, (fee, tgt, dd, fdd, fdll, buf, cap) in {"25K": (109, 1500, 1000, 1000, 500, 1100, 600), "50K": (165, 3000, 2000, 2000, 1000, 2100, 1000),
                                                "100K": (265, 6000, 3000, 2500, 1250, 2600, 1500), "150K": (369, 9000, 4500, 3500, 1750, 3600, 2500)}.items():
    PLANS[f"Tradeify Select Daily {k}"] = plan(fee=fee, target=tgt, dd=dd, eval_cons=0.4, fdd=fdd, fdll=fdll,
                                               payout=dict(kind="daily", buffer=buf, cap=cap, min=250, fresh_mult=2.0))
# ---- MFFU Rapid: eval 50% consistency (EOD); funded intraday trailing locking at +100, daily payouts of everything
#      above the buffer (MLL + 100), min $500, 90/10. Price not found -> use Tradeify Select prices as a placeholder.
for k, (fee, tgt, dd) in {"25K": (109, 1500, 1000), "50K": (165, 3000, 2000), "100K": (265, 6000, 3000), "150K": (369, 9000, 4500)}.items():
    PLANS[f"MFFU Rapid {k}*"] = plan(fee=fee, target=tgt, dd=dd, eval_cons=0.5, f_intraday=True,
                                     max_funded=5 if k in ("25K", "50K") else 3,
                                     payout=dict(kind="buffer", buffer=dd + 100, min=500, every=1))
# ---- MFFU Pro: eval 50% consistency; funded EOD, buffer MLL+100, payout every 14 calendar days (~10 trading), min
#      $1,000, 80/20, floor static at +100 after the first payout. Price placeholder as above.
for k, (fee, tgt, dd) in {"50K": (165, 3000, 2000), "100K": (265, 6000, 3000), "150K": (369, 9000, 4500)}.items():
    PLANS[f"MFFU Pro {k}*"] = plan(fee=fee, target=tgt, dd=dd, eval_cons=0.5, split=0.8,
                                   max_funded=5 if k == "50K" else 3,
                                   payout=dict(kind="buffer", buffer=dd + 100, min=1000, every=10))


def run_eval(m, s, p):
    eq, peak, best, locked = 0.0, 0.0, 0.0, False
    for d in range(s, len(m)):
        thr = 100.0 if locked else peak - p["dd"]
        start = eq
        for v in m[d]:
            if v:
                eq += v
                if eq <= thr:
                    return "fail", d
                if p["eval_dll"] is not None and eq - start <= -p["eval_dll"]:
                    break
        best = max(best, eq - start)
        if eq >= p["target"] and (p["eval_cons"] is None or best <= p["eval_cons"] * eq):
            return "pass", d
        peak = max(peak, eq)
        locked = locked or (p["eval_lock"] and peak >= p["dd"] + 100)
    return None, len(m) - 1


def run_funded(m, s, p):
    po = p["payout"]
    bal, peak, locked, pays = 0.0, 0.0, False, []
    wins, cyc, best_cyc, fresh, last_pay = 0, 0.0, 0.0, 0.0, s - 999
    for d in range(s, min(s + FUNDED_DAYS, len(m))):
        thr = 100.0 if locked else peak - p["fdd"]
        start = bal
        for v in m[d]:
            if v:
                bal += v
                if bal <= thr:
                    return pays, d, "blown"
                if p["f_intraday"] and not locked:
                    peak = max(peak, bal)
                    thr = peak - p["fdd"]
                    if peak >= p["lock_trigger"]:
                        locked, thr = True, 100.0
                if p["fdll"] is not None and bal - start <= -p["fdll"]:
                    break
        day = bal - start
        cyc += day
        fresh += day
        best_cyc = max(best_cyc, day)
        wins += day >= po.get("day", 1e9)
        peak = max(peak, bal)
        locked = locked or peak >= p["lock_trigger"]
        req = 0.0
        if po["kind"] == "windays":
            ok = wins >= po["days"] and cyc > 0 and bal >= po.get("minbal", 0) and \
                (po.get("cons") is None or best_cyc <= po["cons"] * cyc)
            if ok:
                cap = po["caps"][min(len(pays), len(po["caps"]) - 1)] if "caps" in po else po["cap"]
                req = min(po["pct"] * bal, cap, bal - 100)
        elif po["kind"] == "daily":
            if bal > po["buffer"]:
                req = min(po["fresh_mult"] * max(fresh, 0), po["cap"], bal - po["buffer"])
        elif po["kind"] == "buffer":
            if d - last_pay >= po["every"] and bal > po["buffer"]:
                req = bal - po["buffer"]
        req = min(req, bal - 100.0 - p.get("keep", 0.0))  # optional cushion kept above the post-payout floor (+100)
        if req >= max(po["min"], 1.0):
            pays.append((d, p["split"] * req))
            bal -= req
            locked, wins, cyc, best_cyc, fresh, last_pay = True, 0, 0.0, 0.0, 0.0, d
            if p["max_payouts"] and len(pays) >= p["max_payouts"]:
                return pays, d, "moved live"
    return pays, min(s + FUNDED_DAYS, len(m)) - 1, "open"


def program(m, dates, p):
    evals, funded, flows, rows = [], [], [], []
    for s in range(0, len(dates) - 30, STEP):
        evals = [e for e in evals if e >= s]
        funded = [f for f in funded if f >= s]
        if len(evals) >= MAX_EVALS or len(funded) >= p["max_funded"]:
            continue
        res, end = run_eval(m, s, p)
        evals.append(end)
        flows.append((s, -p["fee"]))
        pays = []
        if res == "pass" and len([f for f in funded if f >= end]) < p["max_funded"]:
            pays, fend, _ = run_funded(m, end + 1, p)
            funded.append(fend)
            flows += [(d, x) for d, x in pays]
        rows.append(dict(res=res, days=end - s + 1, n_pay=len(pays), paid=sum(x for _, x in pays)))
    return pd.DataFrame(rows), pd.DataFrame(flows, columns=["di", "usd"])


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    px_now = g.Days("NQ").C[-1, -1]
    hist = {}
    for name, assets in (("2015-2026", ["NQ2015", "NQ"]), ("2010-2014", ["NQ2010"])):
        t, dates = trades(assets)
        hist[name] = (day_matrix(t[t.strat == "noise"], dates, px_now, dict(noise=1)), dates)
    rows = []
    for pname, p in PLANS.items():
        for ratio in (1.5, 2.0, 2.5):
            size = max(1, int(round(ratio * p["dd"] / 1000)))
            for hname, (base, dates) in hist.items():
                ev, fl = program(base * size, dates, p)
                yrs = (dates[-1] - dates[0]).days / 365.25
                yr = fl.assign(y=dates[fl.di].year).groupby("y")["usd"].sum()
                passed = ev.res == "pass"
                rows.append(dict(plan=pname, mnq=size, ratio=ratio, history=hname, fee=p["fee"], evals_per_yr=len(ev) / yrs,
                                 pass_rate=passed.mean(), med_days=ev.loc[passed, "days"].median(),
                                 payouts_per_pass=ev.loc[passed, "n_pay"].mean(), paid_per_eval=ev["paid"].mean(),
                                 net_per_eval=ev["paid"].mean() - p["fee"], net_per_yr=fl["usd"].sum() / yrs,
                                 worst_year=yr.min(), losing_years=f"{(yr < 0).sum()}/{len(yr)}"))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "firm_compare_sim.csv", index=False)
    wide = res.pivot_table(index=["plan", "mnq", "ratio"], columns="history",
                           values=["pass_rate", "paid_per_eval", "net_per_eval", "net_per_yr"], aggfunc="first", sort=False)
    wide.columns = [f"{a}|{b}" for a, b in wide.columns]
    print(wide.round(2).to_string())
