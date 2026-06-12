"""Real-data run — the headline numbers behind Study 05's verdict.

Loads the cached liquid US universe (the parquets Study 04 already cached — total-return
mode, so a dividend on one leg can't masquerade as a divergence), runs the full GGR
pipeline rolling across the sample, and the modern teardown: formation → backtest →
decay-by-year → the bid-ask-bounce wait rule → market-neutrality → cost sweep →
capacity. Cache-only: names with no parquet are skipped, never silently re-downloaded.

    python examples/verify_real.py            # cache-only (the default)
    python examples/verify_real.py --fetch    # (re)build the cache first — SLOW

``--fetch`` walks the Study 04 feed's ticker list and re-downloads each name through
Study 04's own cache writer (one parquet per name, split-only). Use it when the cache
is absent or was written by a pre-fix revision of the data layer (the old split-only
mode double-adjusted across split dates, corrupting every name that ever split —
AAPL, NVDA, ...). Expect a couple hundred Yahoo! calls; delisted names fail and are
skipped, which is the survivorship caveat stated in the README.

Prints every table and writes a reproducible results doc (as-of date + a content
fingerprint of the price inputs) to `docs/results.md`, so the README verdict traces to
an exact run.
"""

import os
import sys

import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from pairs_trading import backtest, data, pairs, robustness
from quantlab.repro import DEFAULT_AS_OF, fingerprint

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

OUT = os.path.join(_STUDY, "docs", "results.md")

# Rule: 12-month formation + 6-month trading, top-20 minimum-SSD pairs, open at 2·sigma,
# close on crossing, honest 1-day execution (wait=1), large-cap cost defaults.
TOP_N, FORM, TRADE, K, WAIT = 20, 252, 126, 2.0, 1

# The cached universe is a *liquid ~170-name basket*, not CRSP. Its breadth grows over
# time — a handful of blue chips in the 1960s, 170+ today — so only the modern era has
# enough names to yield genuinely tight minimum-distance pairs (the regime GGR relied
# on). We headline that era and show the full sample as context, never silently.
MODERN_START = "2004-01-01"     # first trading window ~2005, eligible names >= 60 throughout


def _eligibility(panel: pd.DataFrame) -> pd.DataFrame:
    """Eligible-name count and best/worst selected SSD per formation window."""
    rows = []
    for fs, ts, te in backtest._windows(len(panel), FORM, TRADE):
        form = panel.iloc[fs:ts]
        n_elig = int(sum(form[c].notna().all() for c in form.columns))
        sel = pairs.select_pairs(form, top_n=TOP_N)
        rows.append({
            "trade_start": panel.index[ts].date(),
            "n_eligible": n_elig,
            "best_ssd": sel[0].ssd if sel else float("nan"),
            "worst_ssd": sel[-1].ssd if sel else float("nan"),
        })
    return pd.DataFrame(rows).set_index("trade_start")


def _run_block(panel, dvol):
    res = backtest.run(panel, top_n=TOP_N, form_len=FORM, trade_len=TRADE, k=K, wait=WAIT)
    neutral = robustness.market_neutrality(res.daily, data.market_return(panel))
    boot = robustness.bootstrap_sharpe(res.daily)
    edge_bps = max(res.stats.get("mean_trade_net", 0.0), 1e-9) * 1e4
    cap = robustness.capacity(dvol, res.trades, edge_bps=max(edge_bps, 20.0))
    return res, neutral, boot, cap


def _round4(d: dict) -> dict:
    """Round the numeric values of a stats dict, pass anything else through.

    The quantlab bootstrap dict now carries non-numeric provenance fields
    (e.g. ``method='cbb'``); rounding blindly would crash on them.
    """
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _fetch_cache() -> None:
    """(Re)build the shared Study 04 price cache from the committed WSB feed.

    Reuses Study 04's ``social_oracle.data.fetch`` — the single cache writer for
    these parquets — so the format and the adjustment convention stay canonical.
    Existing parquets are removed first: a cache written by the pre-fix data layer
    holds double-split-adjusted prices and must not survive a refresh.
    """
    study04 = os.path.abspath(os.path.join(_STUDY, "..", "04-social-oracle"))
    sys.path.insert(0, study04)
    from social_oracle import data as oracle_data  # noqa: E402

    feed = pd.read_csv(os.path.join(study04, "_data", "wsb_mentions.csv"))
    tickers = sorted(set(feed["ticker"].astype(str)) | {"SPY"})
    cache_dir = os.path.abspath(data.DEFAULT_CACHE)
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".parquet"):
                os.remove(os.path.join(cache_dir, f))
    print(f"fetching {len(tickers)} names into {cache_dir} (delisted names will fail; that's the survivorship caveat) ...")
    ok, dead = 0, []
    for i, tk in enumerate(tickers, 1):
        try:
            oracle_data.fetch(tk, mode="split_only", use_cache=False)
            ok += 1
        except Exception:
            dead.append(tk)
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} done ({ok} priced, {len(dead)} missing)")
    print(f"fetched {ok}/{len(tickers)} names; missing/delisted ({len(dead)}): {', '.join(dead)}")


def main():
    if "--fetch" in sys.argv[1:]:
        _fetch_cache()
    asof = DEFAULT_AS_OF
    frames = data.load_universe(mode="split_only", min_rows=750, asof=asof)
    panel = data.clean_panel(data.close_panel(frames))   # winsorize bad prints (BMW +6.2M%, ...)
    dvol = data.dollar_volume_panel(frames)
    modern = panel.loc[MODERN_START:]
    print(f"universe: {panel.shape[1]} names with >=750 sessions, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")

    elig = _eligibility(panel)
    first60 = elig[elig.n_eligible >= 60].index.min()
    print(f"eligible names: {elig.n_eligible.min()} (early) -> {elig.n_eligible.max()} (today); "
          f">=60 from {first60}")

    # Full sample (context) and modern era (the fair test) — both fully evaluated.
    res_f, neutral_f, boot_f, cap_f = _run_block(panel, dvol)
    res_m, neutral_m, boot_m, cap_m = _run_block(modern, dvol)
    decay = robustness.decay_by_year(panel, top_n=TOP_N, form_len=FORM, trade_len=TRADE, k=K, wait=WAIT)
    waitfx = robustness.wait_rule_effect(modern, top_n=TOP_N, form_len=FORM, trade_len=TRADE, k=K)
    sweep = backtest.cost_sweep(modern, top_n=TOP_N, form_len=FORM, trade_len=TRADE, k=K, wait=WAIT)

    fp = fingerprint(panel.fillna(0.0), round_dp=4)

    def show(tag, res, neutral, boot, cap):
        print(f"\n[{tag}]")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()})
        print("  neutrality", _round4(neutral))
        print("  bootstrap ", _round4(boot))
        print("  capacity  ", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in cap.items()})

    show("MODERN 2005-2026 (headline)", res_m, neutral_m, boot_m, cap_m)
    show("FULL 1962-2026 (context)", res_f, neutral_f, boot_f, cap_f)
    print("\n[decay by year]\n", decay.round(4).to_string())
    print("\n[wait rule, modern]\n", waitfx.round(4))
    print("\n[cost sweep, modern]\n", sweep.round(4))

    _write_results(OUT, panel, modern, elig, first60,
                   res_m, neutral_m, boot_m, cap_m,
                   res_f, neutral_f, boot_f,
                   decay, waitfx, sweep, asof, fp)
    print(f"\nwrote {OUT}")


def _write_results(path, panel, modern, elig, first60,
                   res_m, neutral_m, boot_m, cap_m,
                   res_f, neutral_f, boot_f,
                   decay, waitfx, sweep, asof, fp):
    def md(df):
        return "```\n" + df.round(4).to_string() + "\n```"

    def stats_line(res):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()}

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Results — Study 05 (Twin-Spread) on the cached liquid universe

*Generated by [`examples/verify_real.py`](../examples/verify_real.py). Universe: the
cached deep-history US names (split-only prices — see the data-mode note in
[`pairs_trading/data.py`](../pairs_trading/data.py)). Rule: {TOP_N} minimum-SSD pairs,
{FORM}-session formation, {TRADE}-session trading, open at {K:.0f}·sigma, close on
crossing, **wait={WAIT}** (honest next-session execution). As-of **{asof}**. Price
fingerprint **`{fp}`** ({panel.shape[1]} names, {panel.index.min().date()} →
{panel.index.max().date()}).*

> **Read the universe before the verdict.** This is a *liquid ~{panel.shape[1]}-name basket*
> (inherited from Study 04's cache), **not** the thousands-of-names CRSP universe GGR
> (1999) formed from. Its breadth grows over time — **{int(elig.n_eligible.min())}** eligible
> names in the early 1960s, **{int(elig.n_eligible.max())}** today — so only the modern era
> has enough names to yield genuinely tight minimum-distance pairs (eligible ≥ 60 from
> **{first60}**). We **headline the modern era** and show the full sample as context. The
> pre-2000 numbers are formed from 7–45 names and are *not* a GGR replication.
> The basket is also **survivor-biased**: the cache keeps only names that could still be
> re-downloaded at fetch time, so delisted losers are missing — a bias that *flatters* a
> convergence rule (the pairs that broke and vanished aren't there to lose money), which
> makes the negative verdict conservative.

## Headline — MODERN era ({modern.index.min().date()} → {modern.index.max().date()}), committed capital, wait=1
`{stats_line(res_m)}`
- market neutrality: `{_round4(neutral_m)}`
- bootstrap Sharpe CI: `{_round4(boot_m)}`
- capacity (per leg): `{ {k: (round(v, 1) if isinstance(v, float) else v) for k, v in cap_m.items()} }`

## Full sample (context only — thin early universe)
`{stats_line(res_f)}`
- bootstrap Sharpe CI: `{_round4(boot_f)}`

## Eligible-name growth + pair tightness (every 4th window)
{md(elig.iloc[::4])}

## Decay by year — the same rule, run year by year (full sample)
{md(decay)}

## Wait rule — GGR's bid-ask-bounce control, modern era (monthly net by execution lag)
{md(waitfx)}

## Cost sweep — modern era (per-leg half-spread bps → committed monthly net)
{md(sweep)}
""")


if __name__ == "__main__":
    main()
