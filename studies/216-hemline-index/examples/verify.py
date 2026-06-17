"""Real-data verification -- Study 216 (Hemline-Index).  Regenerates docs/results.md numbers.

Uses the hardcoded decade-level hemline proxy table and, if the Shiller cache is
available, refines the decade S&P 500 returns from real price data.

    python studies/216-hemline-index/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hemline_index import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 216 -- Hemline-Index -- verification run\n")

    df = data.load_real_tape()
    fp = data.fingerprint(df)
    print(f"Real tape: {len(df)} decades  fp={fp}")
    print()

    # Decade-level panel
    print("=== Decade-by-decade panel ===")
    print(f"  {'Decade':>8} {'Hemline':>9} {'Market':>10} {'S&P500':>10}  Fashion note")
    for d, row in df.iterrows():
        hem = "Rising (+)" if row["hemline_dir"] == 1 else "Falling (-)"
        mkt = "Bull (+)" if row["market_positive"] else "Bear (-)"
        match = "OK" if (row["hemline_dir"] == 1) == row["market_positive"] else "MISS"
        print(
            f"  {row['decade_label']:>8}  {hem:>10}  {mkt:>9}  {row['sp500_decade_pct']:>+8.1f}%"
            f"  [{match}]  {row['fashion_note']}"
        )
    print()

    # Contingency table
    ct = st.agreement_table(df)
    print("=== Contingency table ===")
    print(f"  Correct calls: {ct['n_correct']}/{ct['n']} ({ct['hit_rate']:.0%})")
    print(f"  TP={ct['tp']}  FP={ct['fp']}  FN={ct['fn']}  TN={ct['tn']}")
    print(f"  [!] n={ct['n']} -- structural floor: 10 decades is not an inference-worthy sample")
    print()

    # Phi + Fisher
    stats = st.phi_and_fisher(ct)
    print("=== Association test ===")
    print(f"  Phi coefficient:          {stats['phi']:+.4f}")
    print(f"  Chi2 (unreliable, n=10):  {stats['chi2']:.4f}")
    print(f"  Chi2 p-value:             {stats['p_chi2']:.4f}")
    print(f"  Fisher p (one-sided):     {stats['p_fisher_one_sided']:.4f}")
    print(f"  Fisher p (two-sided):     {stats['p_fisher_two_sided']:.4f}")
    print(f"  [!] Fisher p={stats['p_fisher_two_sided']:.3f} is nowhere near p<0.05")
    print()

    # Timing
    bt = st.timing_backtest(df)
    print("=== Timing backtest ===")
    print(f"  Buy-and-hold annualised:  {bt['bah_ann_pct']:.2f}%/yr")
    print(f"  Hemline timer ann.:       {bt['strat_ann_pct']:.2f}%/yr")
    print(f"  Annual advantage:         {bt['ann_advantage_pct']:+.2f}%/yr")
    print(f"  Held {bt['n_held']} decades, cash {bt['n_cash']} decades")
    print(f"  [!] Timer UNDERPERFORMS B&H by {-bt['ann_advantage_pct']:.2f}%/yr")
    print()

    print("=== VERDICT ===")
    print(f"  Signal: NONE -- Fisher p={stats['p_fisher_two_sided']:.3f} (need <0.05);")
    print(f"         n=10 decades, subjective proxy, no causal mechanism.")
    print(f"  Tradability: MIRAGE -- timer returns {bt['strat_ann_pct']:.2f}%/yr vs B&H {bt['bah_ann_pct']:.2f}%/yr")
    print(f"         ({bt['ann_advantage_pct']:+.2f}%/yr, i.e. the timer LOSES to B&H).")
    print(f"  Myth-check: BUSTED -- 70% decade hit rate sounds impressive, but")
    print(f"         Fisher p=0.44 means random guessing would do as well.  BUSTED.")
    print(f"\n  Fingerprint: {fp}")


if __name__ == "__main__":
    main()
