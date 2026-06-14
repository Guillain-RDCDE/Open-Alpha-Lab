"""Offline deterministic demo — Study 154 (Leverage-Anomaly).

Runs entirely on the synthetic panel (no network, no EDGAR cache needed).
Demonstrates that:
  - With a planted leverage premium the low-minus-high spread is recoverable.
  - Under the null (premium=0) the spread is consistent with zero.
  - A random-portfolio control is close to zero in both cases.

Exits 0.  Deterministic given the seed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leverage_anomaly import data, strategy as st


def _banner(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _show(label: str, stats: dict) -> None:
    print(
        f"  {label:<28s} mean={stats['mean']*100:+.1f}%/yr  "
        f"t={stats['tstat']:+.2f}  Sharpe={stats['sharpe']:+.2f}  "
        f"hit={stats['hit_rate']*100:.0f}%  n={stats['n']}"
    )


def main() -> None:
    print("Study 154 — Leverage-Anomaly  |  synthetic demo  |  seed=154")

    # ------------------------------------------------------------------ #
    # 1. Null panel: premium = 0                                          #
    # ------------------------------------------------------------------ #
    _banner("NULL PANEL (premium=0): no anomaly planted")
    lev0, fwd0, truth0 = data.synthetic_panel(
        n_firms=250, n_years=15, premium=0.0, seed=154
    )
    spread0 = st.quintile_spread(lev0, fwd0)
    _show("low-minus-high", st.summarize(spread0["spread"]))
    exc0 = st.random_portfolio_excess(lev0, fwd0, n_draws=200, seed=154)
    print(f"  random-control excess        mean={exc0.mean()*100:+.2f}%/yr  std={exc0.std()*100:.2f}%")
    print(f"  truth: premium={truth0.premium}  has_premium={truth0.has_premium}")

    # ------------------------------------------------------------------ #
    # 2. Anomaly panel: premium = 0.08                                    #
    # ------------------------------------------------------------------ #
    _banner("ANOMALY PANEL (premium=0.08): low-lev outperforms by design")
    lev1, fwd1, truth1 = data.synthetic_panel(
        n_firms=250, n_years=15, premium=0.08, seed=154
    )
    spread1 = st.quintile_spread(lev1, fwd1)
    _show("low-minus-high", st.summarize(spread1["spread"]))
    exc1 = st.random_portfolio_excess(lev1, fwd1, n_draws=200, seed=154)
    print(f"  random-control excess        mean={exc1.mean()*100:+.2f}%/yr  std={exc1.std()*100:.2f}%")
    print(f"  truth: premium={truth1.premium}  has_premium={truth1.has_premium}")

    # ------------------------------------------------------------------ #
    # 3. Year-by-year breakdown (anomaly panel)                           #
    # ------------------------------------------------------------------ #
    _banner("YEAR-BY-YEAR SPREAD (anomaly panel)")
    print(f"  {'year':>4s}  {'low%':>7s}  {'high%':>7s}  {'spread%':>8s}  {'n':>5s}")
    for y, row in spread1.iterrows():
        print(
            f"  {int(y):>4d}  {row['low']*100:>+7.1f}  {row['high']*100:>+7.1f}  "
            f"{row['spread']*100:>+8.1f}  {int(row['n']):>5d}"
        )

    # ------------------------------------------------------------------ #
    # 4. Fingerprint sanity                                               #
    # ------------------------------------------------------------------ #
    print()
    # Leverage signal is identical (same seed, premium only shifts fwd_ret)
    fp_fwd0 = data.fingerprint(fwd0)
    fp_fwd1 = data.fingerprint(fwd1)
    print(f"  fingerprint(null fwd_ret):    {fp_fwd0}")
    print(f"  fingerprint(anomaly fwd_ret): {fp_fwd1}")
    assert fp_fwd0 != fp_fwd1, "forward-return fingerprints must differ between null and anomaly"
    # Different seeds produce different leverage panels
    lev_alt, _, _ = data.synthetic_panel(n_firms=250, n_years=15, premium=0.0, seed=999)
    fp_alt = data.fingerprint(lev_alt)
    assert data.fingerprint(lev0) != fp_alt, "different seeds must yield different leverage panels"

    print()
    print("Demo completed successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)
