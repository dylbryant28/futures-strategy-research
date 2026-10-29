"""S2 noise-area strategy on NQ: base config (k=1.0, 14-day noise, checks every 30 min, band exit) plus each
refinement/filter applied ONE AT A TIME. Per contract (vol sizing: average 1 contract), net of 1 pt round trip.

Base rules (Zarattini, Aziz & Barbon 2024, as in lit_futures_grid.s2_noise):
  sigma(t) = mean over the previous 14 sessions of |price(t) / open - 1|
  upper = max(open, prev close) * (1 + k sigma(t)), lower = min(open, prev close) * (1 - k sigma(t))
  at each check (10:00, 10:30 ... 15:30): long exits below upper (short above lower); flat -> long above upper /
  short below lower at the check bar's close; flat at 16:00.
"""
import numpy as np
import pandas as pd

from lit_futures_grid import Days
from config import DATA_DIR

FOMC = pd.to_datetime([
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"])


def rolling_pct(x, n=252, minp=60):
    """Percentile of x[i] within the previous n values (no look-ahead)."""
    s = pd.Series(x)
    return s.rolling(n + 1, min_periods=minp + 1).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).to_numpy()


class Ctx:
    def __init__(self, d):
        self.d = d
        C = d.C
        nd = len(C)
        close = C[:, -1]
        ret = np.where(np.isnan(d.prev_close), np.nan, close / d.prev_close - 1)
        self.rv14 = pd.Series(ret).rolling(14, min_periods=10).std().shift(1).to_numpy()
        self.rv_pct = rolling_pct(self.rv14)
        sig14 = pd.DataFrame(np.abs(C / C[:, :1] - 1)).rolling(14, min_periods=14).mean().shift(1).to_numpy()
        self.sig30_pct = rolling_pct(sig14[:, 30])
        cv = np.cumsum(d.V, axis=1)
        slot = np.full_like(cv, np.nan)
        slot[:, 30:] = cv[:, 30:] - cv[:, :-30]
        self.slot_vol = slot
        self.slot_avg = pd.DataFrame(slot).rolling(14, min_periods=10).mean().shift(1).to_numpy()
        self.gap = C[:, 0] - d.prev_close
        dates = d.dates
        first_fri = (dates.weekday == 4) & (dates.day <= 7)
        self.masks = {
            "fomc": dates.normalize().isin(FOMC),
            "nfp": np.asarray(first_fri),
        }
        self.weekday = dates.weekday.to_numpy()
        self.year = dates.year.to_numpy()
        self.nd = nd


def simulate(ctx, k=1.0, lb=14, every=30, offset=0, exit="band", side="both", last_entry=None, gap_adj=True,
             entry="close", stop_mult=None, target_mult=None, sizing="fixed", allow=None, vol_ratio=None, gap_align=None):
    d = ctx.d
    C, H, L, O, VW, M = d.C, d.H, d.L, d.O, d.vwap, d.M
    nd = ctx.nd
    sig = pd.DataFrame(np.abs(C / C[:, :1] - 1)).rolling(lb, min_periods=lb).mean().shift(1).to_numpy()
    pc = d.prev_close if gap_adj else np.full(nd, np.nan)
    hi_base = np.fmax(C[:, 0], np.nan_to_num(pc, nan=-np.inf))
    lo_base = np.fmin(C[:, 0], np.nan_to_num(pc, nan=np.inf))
    U = hi_base[:, None] * (1 + k * sig)
    Lb = lo_base[:, None] * (1 - k * sig)
    checks = set(range(every + offset, M, every))
    first_check = every + offset
    last_entry = M - 1 if last_entry is None else last_entry
    if sizing == "vol":
        size = np.nanmedian(ctx.rv14) / ctx.rv14
        size = np.clip(size, 0.25, 4.0)
    else:
        size = np.ones(nd)
    trades = []
    for i in range(nd):
        if np.isnan(sig[i, 0]) or np.isnan(size[i]) or (allow is not None and not allow[i]):
            continue
        pos, entry_px, stop, tgt, pending, e_col, e_risk = 0, 0.0, None, None, 0, 0, 0.0

        def close_out(px, m):
            trades.append((i, pos, size[i], pos * (px - entry_px), e_col, e_risk, m))

        def can_enter(s, m):
            if m > last_entry or (side == "long" and s < 0) or (side == "short" and s > 0):
                return False
            if gap_align == "with" and not (np.sign(ctx.gap[i]) == s):
                return False
            if gap_align == "against" and not (np.sign(ctx.gap[i]) == -s):
                return False
            if vol_ratio is not None:
                a = ctx.slot_avg[i, m]
                if not (a > 0) or ctx.slot_vol[i, m] / a < vol_ratio:
                    return False
            return True

        def open_pos(s, px, m):
            nonlocal pos, entry_px, stop, tgt, e_col, e_risk
            pos, entry_px, e_col = s, px, m
            width = (U[i, m] - Lb[i, m]) / 2 if not np.isnan(U[i, m]) else 0
            noise = k * sig[i, m] * C[i, 0]
            stop = px - s * stop_mult * noise if stop_mult else None
            e_risk = stop_mult * noise if stop_mult else noise  # points at risk (noise if no stop)
            tgt = px + s * target_mult * noise if target_mult else None

        for m in range(1, M + 1):
            if pending and pos == 0:
                open_pos(pending, O[i, m], m)
                pending = 0
            if pos != 0:
                if stop is not None and ((pos == 1 and L[i, m] <= stop) or (pos == -1 and H[i, m] >= stop)):
                    close_out(min(stop, O[i, m]) if pos == 1 else max(stop, O[i, m]), m); pos = 0
                elif tgt is not None and ((pos == 1 and H[i, m] >= tgt) or (pos == -1 and L[i, m] <= tgt)):
                    close_out(max(tgt, O[i, m]) if pos == 1 else min(tgt, O[i, m]), m); pos = 0
            if m == M:
                break
            if m in checks:
                p = C[i, m]
                if pos == 1 and p < (max(U[i, m], VW[i, m]) if exit == "vwap" else U[i, m]):
                    close_out(p, m); pos = 0
                elif pos == -1 and p > (min(Lb[i, m], VW[i, m]) if exit == "vwap" else Lb[i, m]):
                    close_out(p, m); pos = 0
                if pos == 0 and entry in ("close", "next_open"):
                    s = 1 if p > U[i, m] else (-1 if p < Lb[i, m] else 0)
                    if s and can_enter(s, m):
                        if entry == "close":
                            open_pos(s, p, m)
                        else:
                            pending = s
            if entry == "stop" and pos == 0 and first_check <= m <= last_entry:
                up, dn = H[i, m] >= U[i, m], L[i, m] <= Lb[i, m]
                if up != dn:
                    s = 1 if up else -1
                    if can_enter(s, m):
                        open_pos(s, max(U[i, m], O[i, m]) if up else min(Lb[i, m], O[i, m]), m)
        if pos != 0:
            close_out(C[i, M], M)
    return pd.DataFrame(trades, columns=["day", "side", "size", "gross", "col", "risk", "xcol"])


def metrics(t, ctx, cost):
    if t.empty:
        return dict(trades=0)
    net = (t["gross"] - cost) * t["size"]
    day = np.zeros(ctx.nd)
    np.add.at(day, t["day"].to_numpy(), net.to_numpy())
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    eq = np.cumsum(day)
    by_year = pd.Series(day).groupby(ctx.year).sum()
    return dict(trades=len(t), longs=int((t.side == 1).sum()), win=(net > 0).mean(), pf=gains / losses,
                sharpe=day.mean() / day.std() * np.sqrt(252), avg_pts=net.sum() / t["size"].sum(),
                total_pts=net.sum(), max_dd=(eq - np.maximum.accumulate(eq)).min(),
                yrs_pos=f"{(by_year > 0).sum()}/{len(by_year)}")


if __name__ == "__main__":
    d = Days("NQ")
    ctx = Ctx(d)
    cost = d.cost
    col = d.col
    rv, s30 = ctx.rv_pct, ctx.sig30_pct
    variants = [
        ("0 base", "k=1 lb=14 every=30 band exit", {}),
        ("A2 side", "long only", dict(side="long")),
        ("A2 side", "short only", dict(side="short")),
        *[("A6 check offset", f"checks at +{o} min (e.g. 10:{o:02d})", dict(offset=o)) for o in (5, 10, 15, 20, 25)],
        ("B7 sizing", "vol-target size (14-day daily vol, avg ~1)", dict(sizing="vol")),
        ("B8 vol regime", "trade only if 14d vol > 50th pct (252d)", dict(allow=rv > 0.5)),
        ("B8 vol regime", "trade only if 14d vol > 75th pct", dict(allow=rv > 0.75)),
        ("B8 vol regime", "trade only if 14d vol < 50th pct", dict(allow=rv < 0.5)),
        ("C9 last entry", "no entries after 15:00", dict(last_entry=col(1500))),
        ("C9 last entry", "no entries after 14:30", dict(last_entry=col(1430))),
        ("C9 last entry", "no entries after 14:00", dict(last_entry=col(1400))),
        ("C10 min noise", "skip days with 10:00 noise < 25th pct", dict(allow=s30 >= 0.25)),
        ("C10 min noise", "skip days with 10:00 noise < 33rd pct", dict(allow=s30 >= 0.33)),
        *[("C11 volume", f"last-30-min volume >= {r}x its 14d avg", dict(vol_ratio=r)) for r in (1.0, 1.2, 1.5)],
        ("C12 gap", "only in the gap's direction", dict(gap_align="with")),
        ("C12 gap", "only against the gap", dict(gap_align="against")),
        *[("C13 weekday", f"skip {n}", dict(allow=ctx.weekday != w)) for w, n in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])],
        ("C13 events", "skip FOMC days", dict(allow=~ctx.masks["fomc"])),
        ("C13 events", "skip NFP days (first Friday)", dict(allow=~ctx.masks["nfp"])),
        ("C13 events", "skip FOMC + NFP", dict(allow=~(ctx.masks["fomc"] | ctx.masks["nfp"]))),
        ("D14 entry", "stop order at the band (intrabar, from 10:00)", dict(entry="stop")),
        ("D14 entry", "enter at next bar open after the check", dict(entry="next_open")),
        *[("D15 stop", f"hard stop {s}x noise from entry (every minute)", dict(stop_mult=s)) for s in (0.5, 1.0, 2.0)],
        *[("D16 target", f"profit target {t}x noise from entry", dict(target_mult=t)) for t in (1.0, 2.0, 3.0)],
        *[("E17 k", f"k={k}", dict(k=k)) for k in (0.6, 0.8, 1.2)],
        *[("E18 lookback", f"{lb}-day noise", dict(lb=lb)) for lb in (10, 20, 30)],
        ("E19 bands", "open-only bands (no gap adjustment)", dict(gap_adj=False)),
        ("ref", "VWAP exit", dict(exit="vwap")),
    ]
    rows = []
    base_t = None
    for group, desc, kw in variants:
        t = simulate(ctx, **kw)
        if group == "0 base":
            base_t = t
        rows.append(dict(group=group, variant=desc, **metrics(t, ctx, cost)))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR + "noise_filters_nq.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"NQ {d.dates[0].date()} -> {d.dates[-1].date()}, {len(d.dates)} days")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # A3: base per year and side
    t = base_t.assign(net=(base_t.gross - cost) * base_t["size"], year=ctx.year[base_t.day.to_numpy()])
    g = t.groupby(["year", "side"])["net"].agg(["size", "sum", "mean"]).unstack("side")
    print("\nA3 base by year x side (count / total pts / avg pts):")
    print(g.round(1).to_string())
