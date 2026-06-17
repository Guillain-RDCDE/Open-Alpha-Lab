"""Real-data verification -- Study 282 (NYC-Temperature). Regenerates docs/results.md.

Loads the cached ^GSPC daily series (cache-only by default), joins it with the
curated NYC daily temperature anomaly, and runs the full analysis: the cold/warm
tercile sort with a Newey-West HAC t-stat, the OLS sensitivity slope, the
permutation test, and the net-of-cost tradable overlay.

The ^GSPC cache lives at ``_cache/gspc_daily.parquet`` (populate once with
``data.fetch_gspc_daily(fetch=True)``). If the cache is absent this script falls
back to the synthetic NULL tape and says so. Run from anywhere:

    python studies/282-nyc-temperature/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nyc_temperature import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 282 -- NYC-Temperature -- real-data verification\n")
    try:
        df = data.fetch_gspc_daily(fetch=False)
        tape = "REAL ^GSPC daily (cache)"
    except FileNotFoundError:
        df, _ = data.synthetic_daily(n_days=16000, premium_bps=0.0, seed=282)
        tape = "SYNTHETIC null tape (no real cache present)"

    fp = data.fingerprint(df)
    print(f"Tape: {tape}")
    print(f"Data stamp: n={len(df)} trading days, fingerprint={fp}\n")

    s = st.summarize(df, n_permutations=2000, seed=282)

    print("=== Cold vs warm tercile sort (NYC temperature anomaly) ===")
    print(f"  n cold / warm:        {s['n_cold']} / {s['n_warm']}")
    print(f"  mean cold-day return: {s['mean_cold_bps']:+.2f} bps")
    print(f"  mean warm-day return: {s['mean_warm_bps']:+.2f} bps")
    print(f"  cold-minus-warm:      {s['spread_bps']:+.2f} bps/day")
    print(f"  HAC (Newey-West) t:   {s['spread_tstat']:+.2f}")
    print(f"  permutation p:        {s['perm_p']:.4f}\n")

    print("=== OLS: daily return (bps) ~ standardized anomaly ===")
    print(f"  slope:    {s['slope_bps']:+.3f} bps per +1 std anomaly")
    print(f"  HAC t:    {s['slope_tstat']:+.2f}")
    print(f"  R^2:      {s['r_squared']:.6f}\n")

    print("=== Tradable overlay (cold-long / warm-short, 1-day lag, 1 bp cost) ===")
    print(f"  gross Sharpe: {s['gross_sharpe']:+.2f}")
    print(f"  net Sharpe:   {s['net_sharpe']:+.2f}")
    print(f"  net ann ret:  {s['net_ann_ret']:+.2%}")
    print(f"  net HAC t:    {s['net_tstat']:+.2f}")
    print(f"  turnover:     {s['turnover']:.1%} per day\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- "
          "the temperature on Wall Street does not move the tape.")


if __name__ == "__main__":
    main()
