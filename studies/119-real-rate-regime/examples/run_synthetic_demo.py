"""Offline deterministic demo for Study 119 (Real-Rate-Regime).

Runs entirely from the synthetic tape (no network, no real data required).
Demonstrates the full analysis spine: level sort, momentum sort, timing overlay,
and the positive control (planted effect → detectable regime difference).

    python examples/run_synthetic_demo.py   # exits 0
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from real_rate_regime import data, strategy as st  # noqa: E402


def _add_monthly_ret(df: pd.DataFrame, seed: int = 119) -> pd.DataFrame:
    """Append a synthetic monthly real log-return for the timing overlay demo."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["monthly_real_ret"] = rng.normal(0.004, 0.048, len(df))
    return df


def main() -> None:
    print("=" * 60)
    print("Study 119 — Real-Rate-Regime  (synthetic demo)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Null world: no planted regime effect
    # ------------------------------------------------------------------
    df_null, truth_null = data.synthetic_world(n_months=600, regime_effect=0.0, seed=119)
    df_null = _add_monthly_ret(df_null)
    print(f"\n[Null world] n_months={truth_null['n_months']}, "
          f"regime_effect={truth_null['regime_effect']}")

    clean_null = df_null.dropna(subset=["real_rate", "real_rate_chg", "fwd12_real"])
    ls_null = st.level_sort(clean_null)
    ms_null = st.momentum_sort(clean_null)
    to_null = st.timing_overlay(df_null)

    print("\n  Forward 12m return by real-rate quartile (null):")
    for q, row in ls_null.iterrows():
        print(f"    {q}: {row['fwd_mean']*100:+.1f}% (n={row['n']:.0f})")

    print("\n  Forward 12m return by regime (null):")
    for reg, row in ms_null.iterrows():
        print(f"    {reg}: {row['fwd_mean']*100:+.1f}% (n={row['n']:.0f})")

    strat_s = st.summarize(to_null["strat_ret"])
    bh_s = st.summarize(to_null["bh_ret"])
    print(f"\n  Timing overlay (null):")
    print(f"    Strategy: mean={strat_s['mean_ann']*100:+.2f}%/yr, "
          f"Sharpe={strat_s['sharpe']:+.3f}, t={strat_s['tstat']:+.2f}")
    print(f"    Buy&hold: mean={bh_s['mean_ann']*100:+.2f}%/yr, "
          f"Sharpe={bh_s['sharpe']:+.3f}, t={bh_s['tstat']:+.2f}")
    diff_t = st.diff_tstat(to_null["strat_ret"], to_null["bh_ret"])
    print(f"    Diff t-stat (strat - BH): {diff_t:+.2f}")
    assert abs(strat_s["sharpe"]) < 1.5, "Null world should show no large timing edge"

    # ------------------------------------------------------------------
    # 2. Effect world: planted negative regime effect
    # ------------------------------------------------------------------
    df_eff, truth_eff = data.synthetic_world(n_months=600, regime_effect=-2.0, seed=119)
    df_eff = _add_monthly_ret(df_eff)
    print(f"\n[Effect world] n_months={truth_eff['n_months']}, "
          f"regime_effect={truth_eff['regime_effect']}")

    clean_eff = df_eff.dropna(subset=["real_rate", "real_rate_chg", "fwd12_real"])
    ms_eff = st.momentum_sort(clean_eff)

    print("\n  Forward 12m return by regime (effect world):")
    for reg, row in ms_eff.iterrows():
        print(f"    {reg}: {row['fwd_mean']*100:+.1f}% (n={row['n']:.0f})")

    rising_eff = ms_eff.loc["rising", "fwd_mean"]
    falling_eff = ms_eff.loc["falling", "fwd_mean"]
    assert rising_eff < falling_eff, (
        f"Effect world must show rising < falling (got {rising_eff:.4f} vs {falling_eff:.4f})"
    )
    print("  [OK] Effect world: rising-rate regime has lower forward returns as expected.")

    # ------------------------------------------------------------------
    # 3. Fingerprint
    # ------------------------------------------------------------------
    fp = data.fingerprint(clean_null)
    print(f"\n  Synthetic fingerprint: {fp}")

    # ------------------------------------------------------------------
    # 4. Structural checks
    # ------------------------------------------------------------------
    assert ls_null["n"].sum() == len(clean_null), "Level sort should cover all observations"
    assert set(ms_null.index) == {"rising", "falling"}, "Momentum sort must have both regimes"

    print("\n[OK] All checks passed — synthetic demo complete.")


if __name__ == "__main__":
    main()
