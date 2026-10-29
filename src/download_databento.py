"""
Download 1-minute bars for a CME futures root from Databento (GLBX.MDP3) and convert them to this
project's parquet format (same columns as nq_1min.parquet / nq_1min_extended_et.parquet).

  python download_databento.py MGC                      # estimate cost only, nothing billed
  python download_databento.py MGC --download           # buy + save
  python download_databento.py MGC --start 2019-01-01   # different start date

API key is read from the DATABENTO_API_KEY environment variable (never hard-code it).

Notes:
  - Symbol is the continuous front contract by VOLUME (e.g. MGC.v.0), which follows the active month and
    skips thin listed months. Databento's continuous prices are NOT back-adjusted, so there's a price jump
    at each roll; instrument_id is kept so rolls can be found.
  - Databento stamps each bar at its START; NinjaTrader (and the NQ data the engines were tuned on) stamps
    at the bar's END. Timestamps are shifted +1 minute so session boundaries line up.
  - ohlcv-1m only has bars for minutes with trades; quiet minutes are simply absent (same as the NQ data).
  - The raw .dbn.zst file is kept, so re-processing never costs credits again.
"""
import argparse
from pathlib import Path

import databento as db
import pandas as pd
from config import DATA_DIR

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
OUT_DIR = Path(DATA_DIR + "databento")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="futures root, e.g. MGC, ES, GC, CL")
    p.add_argument("--start", default="2022-12-26")
    p.add_argument("--end", default=None, help="defaults to the latest date Databento has")
    p.add_argument("--roll", default="v", choices=["v", "c", "n"], help="v=volume, c=calendar, n=open interest")
    p.add_argument("--download", action="store_true", help="actually request (and pay for) the data")
    p.add_argument("--tag", default="", help="suffix for the output parquet names, e.g. _2015 -> mgc_2015_1min.parquet")
    args = p.parse_args()

    client = db.Historical()  # reads DATABENTO_API_KEY
    symbol = f"{args.root}.{args.roll}.0"
    end = args.end or client.metadata.get_dataset_range(dataset=DATASET)["end"]

    req = dict(dataset=DATASET, schema=SCHEMA, symbols=[symbol], stype_in="continuous", start=args.start, end=end)
    cost = client.metadata.get_cost(**req)
    print(f"{symbol} {SCHEMA} {args.start} -> {end}: estimated cost ${cost:,.2f}")
    if not args.download:
        print("Cost estimate only -- rerun with --download to buy it.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{args.root}_{SCHEMA}_{args.start}_{str(end)[:10]}.dbn.zst"
    store = client.timeseries.get_range(**req, path=raw_path)
    print(f"Saved raw data to {raw_path}")

    # Keep UTC order; sorting on local wall-clock time would scramble the repeated hour at the DST fall-back.
    df = store.to_df().sort_index()
    bar_close_utc = df.index + pd.Timedelta(minutes=1)
    out = pd.DataFrame({
        "datetime_pt": bar_close_utc.tz_convert("America/Los_Angeles").tz_localize(None),
        "open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(), "close": df["close"].to_numpy(),
        "volume": df["volume"].to_numpy().astype("int64"),
        "instrument_id": df["instrument_id"].to_numpy(),
    })
    # Match the NQ files' microsecond precision; some engines' datetime math breaks on nanoseconds.
    out["datetime_pt"] = out["datetime_pt"].astype("datetime64[us]")

    pt_path = Path(DATA_DIR + f"{args.root.lower()}{args.tag}_1min.parquet")
    out.to_parquet(pt_path, index=False)
    et = out.rename(columns={"datetime_pt": "datetime_et"})
    et["datetime_et"] = bar_close_utc.tz_convert("America/New_York").tz_localize(None).astype("datetime64[us]")
    et.to_parquet(pt_path.with_name(f"{args.root.lower()}{args.tag}_1min_et.parquet"), index=False)

    rolls = (out["instrument_id"] != out["instrument_id"].shift()).sum() - 1
    print(f"{len(out):,} bars, {out['datetime_pt'].min()} -> {out['datetime_pt'].max()}, {rolls} contract rolls")
    print(out.head(3).to_string(index=False))
    print(f"Wrote {pt_path.name} and {pt_path.stem}_et.parquet")


if __name__ == "__main__":
    main()
