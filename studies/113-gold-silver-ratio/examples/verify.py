"""Real-tape verification — Study 113 (Gold-Silver-Ratio). Regenerates docs/results.md numbers.

Fetches (or reads from cache) GLD and SLV daily bars, computes the gold/silver ratio,
runs the z-score reversion strategy vs a random-direction control, tests cointegration
and unit-roots, estimates the half-life, and sweeps costs. Network is touched only with --fetch.

    python studies/113-gold-silver-ratio/examples/verify.py            # cache-only
    python studies/113-gold-silver-ratio/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gold_silver_ratio import data, strategy as st  # noqa: E402

GOLD_TICKER = data.GOLD_TICKER     # GLD
SILVER_TICKER = data.SILVER_TICKER  # SLV


def main(fetch: bool) -> None:
    print(f"=== Study 113 — Gold-Silver-Ratio ({'fetch' if fetch else 'cache'}) ===\n")

    gold = data.fetch_daily(GOLD_TICKER, fetch=fetch)
    silver = data.fetch_daily(SILVER_TICKER, fetch=fetch)

    if fetch:
        print(f"{GOLD_TICKER}   {gold.index[0].date()}..{gold.index[-1].date()} "
              f"n={len(gold)} fp={data.fingerprint(gold)}")
        print(f"{SILVER_TICKER}   {silver.index[0].date()}..{silver.index[-1].date()} "
              f"n={len(silver)} fp={data.fingerprint(silver)}\n")

    ratio = st.compute_ratio(gold, silver)
    print(f"Ratio: {ratio.index[0].date()} to {ratio.index[-1].date()}, "
          f"n={len(ratio)}, mean={ratio.mean():.1f}, "
          f"min={ratio.min():.1f}, max={ratio.max():.1f}\n")

    # Stationarity of the ratio
    adf_p = st.adf_pvalue(ratio)
    hl = st.half_life(ratio)
    ci_p = st.engle_granger_pvalue(gold["close"], silver["close"])
    print(f"ADF p-value (ratio): {adf_p:.4f}  (< 0.05 = stationary / mean-reverting)")
    print(f"Estimated half-life: {hl:.1f} trading days")
    print(f"Engle-Granger cointegration p-value: {ci_p:.4f}  (< 0.05 = cointegrated)\n")

    # Strategy vs random control
    entries = st.zscore_entries(ratio, window=252, enter_z=2.0, exit_z=0.5)
    print(f"=== z-score strategy (enter |z|>2, exit |z|<0.5, window=252) ===")
    print(f"Trades: n={len(entries)}\n")

    ledger = st.run_trades(gold, silver, entries, cost_bps=0.0)
    rand_ent = st.random_entries(entries, seed=113)
    rand_led = st.run_trades(gold, silver, rand_ent, cost_bps=0.0)

    s = st.summarize(ledger, "ret_gross")
    r = st.summarize(rand_led, "ret_gross")
    print(f"STRATEGY  n={s['n_trades']:4d}  win={s['win_rate']:.3f}  "
          f"mean={s['mean_bps']:+.1f}bps  t={s['tstat']:+.2f}")
    print(f"RANDOM    n={r['n_trades']:4d}  win={r['win_rate']:.3f}  "
          f"mean={r['mean_bps']:+.1f}bps  t={r['tstat']:+.2f}")
    print(f"delta (strategy - random): {s['mean_bps'] - r['mean_bps']:+.1f} bps\n")

    print("=== cost sweep (strategy, net) ===")
    for c in (0.0, 5.0, 10.0, 20.0):
        sn = st.summarize(st.run_trades(gold, silver, entries, cost_bps=c), "ret_net")
        print(f"cost={c:5.1f}bps  net mean={sn['mean_bps']:+.1f}bps  t={sn['tstat']:+.2f}")

    print("\n=== buy-and-hold comparison ===")
    g_ret = st.bah_return(gold)
    s_ret = st.bah_return(silver)
    g_ann = (1 + g_ret) ** (252 / len(gold["close"].dropna())) - 1
    s_ann = (1 + s_ret) ** (252 / len(silver["close"].dropna())) - 1
    print(f"GLD  total log-return = {g_ret*100:+.1f}%  (~{g_ann*100:+.1f}%/yr)")
    print(f"SLV  total log-return = {s_ret*100:+.1f}%  (~{s_ann*100:+.1f}%/yr)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
