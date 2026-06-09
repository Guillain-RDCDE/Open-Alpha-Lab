"""Real-data run — does dealer gamma call the day's character, or is it the VIX in a trenchcoat?

Builds the regime panel on the real tape and runs the teardown:

  * **GEX** at each prior close from the real SPY option chain (Alpha Vantage HISTORICAL_OPTIONS,
    open-interest x gamma under the dealer convention) -> sign per session;
  * **character** of the next session from daily SPY OHLC (range vol + directional efficiency);
  * **VIX** at the prior close as the confound the verdict turns on.

    # HISTORICAL_OPTIONS is a PREMIUM Alpha Vantage endpoint (the free key is rejected); a paid
    # chain source is required for a reliable historical GEX -- this is why Study 14 ships
    # pre-registered. With a premium key (or any equivalent chain API wired into fetch_chain):
    setx ALPHAVANTAGE_API_KEY <key>          # (Windows)   export on *nix
    python examples/verify.py --fetch        # fetch daily bars + up to --max-fetch chains
    # re-run on later days to extend the chain cache, then offline:
    python examples/verify.py                # cache-only: rebuild the panel, no network

Network lives only behind `--fetch`. One request is one trading day's chain, so the daily panel is
accumulated over several runs — each run fetches at most `--max-fetch` chains, evenly spaced across
the range, and says so out loud (house rule: no silent caps). The headline is pinned with
`quantlab.repro.as_of` and stamped with a content fingerprint.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from gamma_gospel import data, decompose
from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

OUT = os.path.join(_STUDY, "docs", "results.md")
CACHE = data.DEFAULT_CACHE
FP_COLS = ["gex", "vix", "rv", "de"]
SYMBOL = "SPY"


def _load_daily(ticker, start, end, fetch):
    """Daily OHLC for ``ticker``, cached to parquet; cache-only unless ``fetch``."""
    safe = ticker.replace("^", "_")
    path = os.path.join(CACHE, f"bars_{safe}_1d.parquet")
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, interval="1d",
                      auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.title)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    raw.index.name = "date"
    os.makedirs(CACHE, exist_ok=True)
    raw.to_parquet(path)
    return raw


def _gather_chains(dates, fetch, max_fetch, delay=13.0):
    """Return {date_str: chain} from cache, fetching up to ``max_fetch`` *new* dates if ``fetch``.

    Always loads every cached date. When fetching, it spends the day's budget on an **evenly-spaced
    sample** of the still-uncached dates across the whole range — so a first 25-chain run spans
    several VIX/GEX regimes rather than 25 consecutive (one-regime) sessions — and throttles each
    call by ``delay`` seconds to respect Alpha Vantage's per-minute free-tier limit.
    """
    import time

    keys = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    chains, missing = {}, []
    for key in keys:
        if os.path.exists(data._chain_cache_path(SYMBOL, key, CACHE)):
            chains[key] = data.fetch_chain(SYMBOL, key, fetch=False)
        else:
            missing.append(key)

    fetched = 0
    if fetch and missing:
        if len(missing) > max_fetch:                         # evenly-spaced pick across the range
            idx = np.linspace(0, len(missing) - 1, max_fetch).round().astype(int)
            todo = [missing[i] for i in sorted(set(idx))]
        else:
            todo = missing
        for i, key in enumerate(todo):
            ch = data.fetch_chain(SYMBOL, key, fetch=True)
            if not ch.empty:
                chains[key] = ch
                fetched += 1
            if delay and i < len(todo) - 1:
                time.sleep(delay)

    print(f"chains: {len(chains) - fetched} cached, {fetched} newly fetched, "
          f"{max(0, len(missing) - fetched)} still uncached "
          f"(premium endpoint; re-run --fetch on later days to fill in)")
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="populate caches (network)")
    ap.add_argument("--start", default="2023-06-01", help="first chain date (YYYY-MM-DD)")
    ap.add_argument("--end", default=DEFAULT_AS_OF, help="last date (YYYY-MM-DD)")
    ap.add_argument("--max-fetch", type=int, default=25, help="max NEW chains to fetch this run")
    args = ap.parse_args()

    spy = _load_daily(SYMBOL, args.start, args.end, args.fetch)
    vixbars = _load_daily("^VIX", args.start, args.end, args.fetch)
    if spy.empty or vixbars.empty:
        print("[skip] no daily-bar cache — run once with --fetch to populate SPY/VIX bars.")
        return

    spots = spy["Close"]
    vix = vixbars["Close"]
    gex_dates = [d for d in spy.index if args.start <= d.strftime("%Y-%m-%d") <= args.end][:-1]
    chains = _gather_chains(gex_dates, args.fetch, args.max_fetch)
    if not chains:
        print("[skip] no chains cached yet — run with --fetch (and an Alpha Vantage key).")
        return

    panel = data.build_panel(chains, spots, vix=vix, bars=spy)
    panel = as_of(panel, DEFAULT_AS_OF)
    if len(panel) < 30:
        print(f"[warn] only {len(panel)} sessions in the panel — fetch more chains before quoting "
              "(the verdict needs a powered sample).")

    fp = fingerprint(panel, cols=FP_COLS)
    neg = panel["neg_gamma"].astype(bool)
    s = decompose.summary(panel)
    print(f"\n=== {SYMBOL} regime panel: {len(panel)} sessions, "
          f"{int(neg.sum())} negative-gamma ({neg.mean():.0%}), fingerprint {fp} ===")
    print(f"VIX on neg-gamma days {panel.loc[neg,'vix'].mean():.1f} vs "
          f"pos-gamma {panel.loc[~neg,'vix'].mean():.1f}")
    for y, name in (("rv", "range vol"), ("de", "directional efficiency")):
        raw, p = s[y]["raw"], s[y]["partial"]
        print(f"[{name}] raw gap {raw['gap']:+.4f} (HAC t {raw['t']:+.1f})  ->  "
              f"survives VIX-control {p['surviving_coef']:+.4f} (HAC t {p['surviving_t']:+.1f}), "
              f"share kept {p['survival_share']:.0%}, dR2 {p['delta_r2']:+.3f}")

    _write_results(OUT, SYMBOL, panel, s, fp, DEFAULT_AS_OF)
    print(f"\nwrote {OUT}")


def _write_results(path, symbol, panel, s, fp, asof):
    neg = panel["neg_gamma"].astype(bool)
    lo, hi = panel.index.min().date(), panel.index.max().date()

    def block(y, name):
        raw, p = s[y]["raw"], s[y]["partial"]
        verdict = ("**survives** — the GEX sign adds real information over the VIX level"
                   if abs(p["surviving_t"]) > 2 and p["survival_share"] > 0.3
                   else "**collapses** — once VIX is partialled out, the GEX sign adds nothing "
                        "(the volatility regime in a trenchcoat)")
        return f"""
## {name} — does negative gamma call it, or is it VIX?

- **Raw gap** (neg-gamma minus pos-gamma): **{raw['gap']:+.4f}** — mean {raw['mean_neg']:.4f} vs
  {raw['mean_pos']:.4f}, HAC *t* = **{raw['t']:+.1f}** on {raw['n']} sessions. The pitch's headline
  effect {'is present' if raw['t'] > 2 else 'is weak/absent'} in the raw data.
- **After controlling for the prior-close VIX**: the surviving negative-gamma coefficient is
  **{p['surviving_coef']:+.4f}** (HAC *t* = **{p['surviving_t']:+.1f}**), i.e. **{p['survival_share']:.0%}**
  of the raw gap. Incremental R² over a VIX-only model: **{p['delta_r2']:+.3f}**.
- **Read:** the effect {verdict}."""

    lines = [f"""# Results — Study 14 (Gamma-Gospel) on real {symbol} options + SPY/VIX

*Generated by [`examples/verify.py`](../examples/verify.py). GEX is computed at each prior close
from the real {symbol} option chain (Alpha Vantage HISTORICAL_OPTIONS: open-interest x gamma, calls
long / puts short — the SqueezeMetrics dealer convention), and attributed to the next session,
whose **range vol** and **directional efficiency** are read from daily {symbol} OHLC. The
prior-close VIX is the confound. As-of **{asof}**; match the fingerprint below to confirm you hold
the same tape.*

## Data stamp

- **{symbol}**: {lo} → {hi}, {len(panel)} sessions, {int(neg.sum())} negative-gamma
  ({neg.mean():.0%}), fingerprint `{fp}`
- VIX on negative-gamma days **{panel.loc[neg,'vix'].mean():.1f}** vs positive-gamma
  **{panel.loc[~neg,'vix'].mean():.1f}** — the confound, in one line: the "amplifier" regime is
  just the high-VIX regime.
{block('rv', 'Range volatility')}
{block('de', 'Directional efficiency (trend vs chop)')}
"""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
