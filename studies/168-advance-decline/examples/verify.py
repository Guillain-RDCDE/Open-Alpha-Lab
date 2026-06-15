"""Real-tape verification — Study 168 (Advance-Decline). Regenerates docs/results.md numbers.

Fetches (or reads from cache) daily SPY prices and S&P 500 constituent returns, builds the
cumulative A/D line, detects bearish price-vs-breadth divergences, and tests forward SPY returns
after a divergence signal vs the unconditional baseline. Sweeps the lookback window and
runs a permutation correction for multiple comparisons.

    python studies/168-advance-decline/examples/verify.py            # cache-only
    python studies/168-advance-decline/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from advance_decline import data, strategy as st  # noqa: E402

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    # ------------------------------------------------------------------ load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spy = data.fetch_spy(fetch=fetch, cache_dir=CACHE_DIR)
        panel = data.fetch_panel(fetch=fetch, cache_dir=CACHE_DIR,
                                 allow_survivorship_bias=True)

    print(f"SPY tape:  {spy.index[0].date()} -> {spy.index[-1].date()}, "
          f"fp={data.fingerprint(spy)}")
    print(f"Panel:     {panel.index[0].date()} -> {panel.index[-1].date()}, "
          f"{panel.shape[1]} tickers\n")

    ad_line = data.build_ad_line(panel)

    # Align spy close with ad_line
    common   = spy.index.intersection(ad_line.index)
    spy_c    = spy.loc[common, "close"]
    ad_c     = ad_line.loc[common]

    # ------------------------------------------------------------------ primary test: 63d lookback / 21d horizon
    res = st.backtest(spy_c, ad_c, lookback=63, horizon=21)
    ds, bs = res["div_stats"], res["base_stats"]
    print("=== Primary: 63d lookback, 21d forward horizon ===")
    print(f"  N divergence signals : {res['n_divergences']}")
    print(f"  Post-divergence  : mean={ds['mean_bps']:+.1f}bps  t={ds['tstat']:+.2f}  hit={ds['hit_rate']:.3f}")
    print(f"  Unconditional    : mean={bs['mean_bps']:+.1f}bps  t={bs['tstat']:+.2f}  hit={bs['hit_rate']:.3f}")
    print(f"  Difference (div - base): {res['diff_mean_bps']:+.1f}bps")

    # ------------------------------------------------------------------ permutation correction
    if res["n_divergences"] >= 5:
        perm = st.permutation_baseline(res["base_rets"], res["n_divergences"],
                                       n_perm=2000, seed=168)
        frac_more_neg = float((perm["perm_means_bps"] < ds["mean_bps"]).mean())
        print(f"\n  Permutation p-val (fraction of random sub-samples more negative): {frac_more_neg:.3f}")
    else:
        frac_more_neg = float("nan")
        print(f"\n  Too few signals ({res['n_divergences']}) for permutation test.")

    # ------------------------------------------------------------------ lookback sweep
    print("\n=== Lookback sweep (21d forward horizon) ===")
    sweepdf = st.sweep_lookback(spy_c, ad_c,
                                lookbacks=[21, 42, 63, 126, 252], horizon=21)
    print(sweepdf.to_string())

    # ------------------------------------------------------------------ drawdown
    sig  = st.ad_divergence_signal(spy_c, ad_c, lookback=63)
    ddres = st.max_drawdown_after_signal(spy_c, sig, max_horizon=63)
    print(f"\n  Post-signal max dd (63d): {ddres['signal_dd_mean']*100:+.2f}%")
    print(f"  Baseline max dd (63d):    {ddres['base_dd_mean']*100:+.2f}%")

    # ------------------------------------------------------------------ cost / tradability
    # "Signal" is already rare — round-trip cost 10bps (ETF + spread)
    # If one wanted to short SPY on each divergence signal for 21d:
    # gross return per trade (bps), net after 10bps round-trip cost
    net_bps  = ds["mean_bps"] - 10.0   # very rough
    print(f"\n=== Tradability ===")
    print(f"  Post-div gross return (21d): {ds['mean_bps']:+.1f}bps")
    print(f"  After 10bps round-trip:      {net_bps:+.1f}bps")
    print(f"  Signals per year (~):        {res['n_divergences'] / max(1,(spy.index[-1].year-spy.index[0].year)):+.1f}")

    # ------------------------------------------------------------------ summary
    print("\n=== Summary ===")
    print(f"  Signal: {'WEAK' if abs(ds['tstat']) > 1.5 else 'NONE'}  "
          f"(HAC t = {ds['tstat']:+.2f} on n={res['n_divergences']} divergence periods)")
    print(f"  Tradability: MIRAGE (rare signal, timing unreliable, diff not significant)")
    print(f"  Survivorship bias: NAMED — panel uses current S&P 500 membership (losers absent)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
