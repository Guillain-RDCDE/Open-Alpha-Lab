"""Real-tape verification -- Study 255 (Fear-Greed). Regenerates docs/results.md numbers.

Joins the curated weekly CNN Fear & Greed series with real ^GSPC weekly closes
(repo cache), bins forward returns by sentiment band, builds the contrarian
fear-minus-greed overlay, pins it against a shuffled-sentiment null and the
unconditional drift, sweeps sub-periods and costs, and runs the synthetic
positive control.  Network is touched only with --fetch.

    python studies/255-fear-greed-index/examples/verify.py          # cache-only
    python studies/255-fear-greed-index/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fear_greed_index import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def main(fetch: bool) -> None:
    panel = data.real_panel(fetch=fetch)
    fr = st.forward_returns(panel)
    print(f"Panel: {len(panel)} weeks ({len(fr)} with forward return)")
    print(f"Range: {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(panel)}")
    print(f"Unconditional forward return: {fr['fwd_ret'].mean()*52*100:+.2f}%/yr\n")

    print("=== Forward return by Fear & Greed band ===")
    rm = st.regime_means(fr)
    for lbl, row in rm.iterrows():
        print(f"  {lbl:14s} n={int(row['n']):4d}  {row['mean_ann']*100:+7.2f}%/yr  "
              f"HAC t={row['tstat']:+5.2f}  hit={row['hit']:.3f}")

    spread = st.spread_returns(fr)
    s_sp = st.summarize(spread)
    s_sp_a = st.annualise_weekly(s_sp)
    print("\n=== Contrarian fear-minus-greed spread ===")
    print(f"Mean: {s_sp['mean']*10000:+.2f} bps/wk = {s_sp['mean']*52*100:+.2f}%/yr  "
          f"HAC t={s_sp['tstat']:+.2f}  hit={s_sp['hit_rate']:.3f}  SR(ann)={s_sp_a.get('sharpe_ann', float('nan')):+.2f}")

    fe = st.long_only_fear_excess(fr)
    s_fe = st.summarize(fe)
    print(f"Long-only fear excess: {s_fe['mean']*52*100:+.2f}%/yr  HAC t={s_fe['tstat']:+.2f}  n={s_fe['n']}")

    sl = st.sentiment_slope(fr)
    print(f"Linear sentiment slope: {sl['slope']*10000:+.2f} bps per 50 F&G pts  HAC t={sl['tstat']:+.2f}")

    if _HAS_QUANTLAB:
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=52, seed=255)
        print(f"\nSpread Sharpe (ann) 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  "
              f"({ci['frac_negative']*100:.0f}% of resamples negative)")

    print("\n=== Sub-period analysis (contrarian spread) ===")
    for label, a, b in [("2011-2016", "2011", "2016"),
                        ("2017-2021", "2017", "2021"),
                        ("2022-2026", "2022", "2026")]:
        sub = spread[a:b]
        s = st.summarize(sub)
        print(f"  {label}: {s['mean']*52*100:+.2f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")

    ts = st.turnover_stats(fr)
    print("\n=== Turnover and costs ===")
    print(f"States: long {ts['pct_long']:.1%}, flat {ts['pct_flat']:.1%}, short {ts['pct_short']:.1%}; "
          f"{ts['n_trades']} trades ({ts['trades_per_year']:.1f}/yr)")
    net = st.net_spread_returns(fr, one_way_bps=5.0, borrow_bps_ann=50.0)
    s_net = st.summarize(net)
    print(f"Net spread @5bps one-way + 50bps/yr borrow: {s_net['mean']*52*100:+.2f}%/yr  HAC t={s_net['tstat']:+.2f}")

    print("\nComputing shuffled-sentiment null (n_draws=2000)...")
    null = st.random_timing_null(fr, n_draws=2000, seed=255)
    pctile = (null < spread.mean()).mean()
    print(f"Null spread mean: {null.mean()*10000:+.2f} bps/wk (std {null.std()*10000:.2f}); "
          f"real spread sits at the {pctile*100:.1f}th percentile")

    print("\n=== Positive control (synthetic) ===")
    for prem in [0.0, 60.0, 120.0]:
        d, _ = data.synthetic_panel(contrarian_bps=prem, seed=255)
        f = st.forward_returns(d)
        sc = st.summarize(st.spread_returns(f))
        print(f"  premium={prem:6.1f} bps -> spread {sc['mean']*10000:+.2f} bps/wk  HAC t={sc['tstat']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
