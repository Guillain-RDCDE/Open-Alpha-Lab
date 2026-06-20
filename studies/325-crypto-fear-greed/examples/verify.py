"""Real-tape verification -- Study 325 (Crypto-Fear-Greed). Regenerates docs/results.md.

Builds the price-derived Fear & Greed proxy gauge from real BTC-USD daily closes
(cache-first), bins forward returns by sentiment band, builds the contrarian
fear-minus-greed overlay, pins it against a shuffled-gauge null, a block-bootstrap
Sharpe CI, sub-periods and a crypto-realistic cost sweep, and runs the synthetic
positive control. Network is touched only with --fetch.

    python studies/325-crypto-fear-greed/examples/verify.py          # cache-only
    python studies/325-crypto-fear-greed/examples/verify.py --fetch  # refresh BTC-USD
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_fear_greed import data, strategy as st  # noqa: E402

# Drop the partial current month so the as-of is never a future/partial bar.
AS_OF = pd.Timestamp("2026-05-31")


def main(fetch: bool) -> None:
    close = data.load_real(fetch=fetch)
    if close is None:
        print("No BTC cache and fetch=False -- cannot regenerate the real run offline.")
        return
    close = close[close.index <= AS_OF]
    panel = data.panel_from_close(close)
    fr = st.forward_returns(panel)
    print(f"BTC tape: {close.index[0].date()} -> {close.index[-1].date()} ({len(close)} days)")
    print(f"Gauge-panel rows: {len(panel)} ({len(fr)} with forward return)")
    print(f"Fingerprint (close): {data.fingerprint(close)}")
    print(f"Unconditional forward return: {fr['fwd_ret'].mean()*365*100:+.2f}%/yr\n")

    print("=== Forward daily return by Fear & Greed band ===")
    rm = st.regime_means(fr)
    for lbl, row in rm.iterrows():
        print(f"  {lbl:14s} n={int(row['n']):5d}  {row['mean_ann']*100:+9.2f}%/yr  "
              f"HAC t={row['tstat']:+5.2f}  hit={row['hit']:.3f}")

    spread = st.spread_returns(fr)
    s_sp = st.summarize(spread)
    s_a = st.annualise(s_sp)
    print("\n=== Contrarian fear-minus-greed spread ===")
    print(f"Mean: {s_sp['mean']*1e4:+.2f} bps/d = {s_sp['mean']*365*100:+.2f}%/yr  "
          f"HAC t={s_sp['tstat']:+.2f}  hit={s_sp['hit_rate']:.3f}  SR(ann)={s_a.get('sharpe_ann', float('nan')):+.2f}")

    ci = st.block_bootstrap_sharpe_ci(spread, n_boot=2000, seed=325)
    print(f"Block-bootstrap Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  "
          f"({ci['frac_negative']*100:.0f}% of resamples negative)")

    fe = st.long_only_fear_excess(fr)
    s_fe = st.summarize(fe)
    print(f"Long-only fear excess: {s_fe['mean']*365*100:+.2f}%/yr  HAC t={s_fe['tstat']:+.2f}  n={s_fe['n']}")

    sl = st.sentiment_slope(fr)
    print(f"Linear sentiment slope: {sl['slope']*1e4:+.2f} bps per 50 gauge pts  HAC t={sl['tstat']:+.2f}")

    print("\n=== Sub-period analysis (contrarian spread) ===")
    for label, a, b in [("2016-2019", "2016", "2019"), ("2020-2022", "2020", "2022"),
                        ("2023-2026", "2023", "2026")]:
        s = st.summarize(spread[a:b])
        print(f"  {label}: {s['mean']*365*100:+.2f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")

    ts = st.turnover_stats(fr)
    print("\n=== Turnover and costs ===")
    print(f"States: long {ts['pct_long']:.1%}, flat {ts['pct_flat']:.1%}, short {ts['pct_short']:.1%}; "
          f"{ts['n_trades']} trades ({ts['trades_per_year']:.1f}/yr)")
    net = st.net_spread_returns(fr, one_way_bps=10.0, borrow_bps_ann=1000.0)
    s_net = st.summarize(net)
    print(f"Net spread @10bps one-way + 1000bps/yr funding: {s_net['mean']*365*100:+.2f}%/yr  HAC t={s_net['tstat']:+.2f}")

    print("\nComputing shuffled-gauge null (n_draws=2000)...")
    null = st.random_timing_null(fr, n_draws=2000, seed=325)
    pctile = (null < spread.mean()).mean()
    print(f"Null spread mean: {null.mean()*1e4:+.2f} bps/d (std {null.std()*1e4:.2f}); "
          f"real spread sits at the {pctile*100:.1f}th percentile")

    print("\n=== Positive control (synthetic) ===")
    for rev in [0.0, 0.2, 0.35, 0.5]:
        d, _ = data.synthetic_btc(reversion=rev, seed=325)
        f = st.forward_returns(data.panel_from_close(d["close"]))
        sc = st.summarize(st.spread_returns(f))
        print(f"  reversion={rev:.2f} -> spread {sc['mean']*1e4:+.2f} bps/d  HAC t={sc['tstat']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(close)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
