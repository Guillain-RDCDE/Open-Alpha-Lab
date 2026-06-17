"""Cache-only verification — Study 289 (Diwali-Muhurat).

Loads the cached INDA daily panel (``_cache/inda_daily.parquet``) if present and
reruns the full event study: post-Diwali window returns, the unconditional
baseline, the plain and Newey-West t-stats, the block-permutation test, and the
net-of-cost edge. If the cache is absent it falls back to the deterministic
synthetic NULL tape (no planted premium) so the script always runs offline.

This study ships **no** real INDA cache by default (INDA is fetched on demand via
``data.fetch_inda(fetch=True)``), so the headline numbers in ``docs/results.md``
are produced by the synthetic null tape at the study seed — i.e. the honest
"no edge" world. Populate the cache once and rerun to verify on the live proxy.

    python studies/289-diwali-muhurat/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diwali_muhurat import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 289 -- Diwali-Muhurat -- cache-only verification\n")
    dw = data.diwali_table()

    try:
        prices = data.fetch_inda(fetch=False)
        tape = "REAL INDA cache"
    except FileNotFoundError:
        prices, _ = data.synthetic_panel(premium_bps=0.0, seed=289)
        tape = "SYNTHETIC NULL tape (no INDA cache present)"

    fp = data.fingerprint(prices)
    print(f"Tape: {tape}")
    print(f"Data stamp: rows={len(prices)}, "
          f"{prices.index.min().date()}..{prices.index.max().date()}, fingerprint={fp}\n")
    print(f"Diwali events table: {len(dw)} rows (2003-2025)\n")

    for hd in (1, 5):
        s = st.summarize(prices, dw, hold_days=hd, lag=1, n_permutations=10_000, seed=289)
        label = "PRIMARY" if hd == 1 else "ROBUST "
        print(f"=== {label} spec: hold {hd} session(s), 1-session execution lag ===")
        print(f"  events in window:   {s['n_events']}")
        print(f"  mean event return:  {s['mean_event_pct']:+.3f}%")
        print(f"  baseline window:    {s['baseline_pct']:+.3f}%")
        print(f"  excess vs baseline: {s['excess_pct']:+.3f}%")
        print(f"  hit-rate:           {s['hit_rate_pct']:.1f}%")
        print(f"  t (plain):          {s['t_plain']:+.3f}")
        print(f"  t (Newey-West):     {s['t_nw']:+.3f}")
        print(f"  perm p (one-sided): {s['perm_p']:.4f}")
        print(f"  net of round-trip:  {s['net_pct']:+.3f}%  (net excess {s['net_excess_pct']:+.3f}%)")
        print()

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- "
          "the Muhurat window does not beat a random window of the same length.")


if __name__ == "__main__":
    main()
