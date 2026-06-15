"""Offline, deterministic demo — Study 203 (Golden-Butterfly).

No network. Builds a synthetic five-asset world and shows the study's spine in
one screen: the Golden Butterfly improves risk-adjusted return (Sharpe) over 100%
stocks *when* a regime cycle is planted — but the gain dissolves on a no-cycle
i.i.d. world.  The regime knob is the only forecastable structure in the tape.

Run:
    python studies/203-golden-butterfly/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from golden_butterfly import data, strategy as st  # noqa: E402

# Synthetic column names map to real-world ETF names
SYNTH_MAP = {"LCG": "SPY", "SCV": "IWN", "BOND": "TLT", "CASH": "SHY", "GOLD": "GLD"}


def _run(cycle_strength: float) -> None:
    frame, _ = data.synthetic_five_asset(n_years=20, cycle_strength=cycle_strength, seed=203)
    ret = st.to_returns(frame).rename(columns=SYNTH_MAP)
    rf = ret["SHY"]

    gb = st.golden_butterfly(ret)
    pp = st.permanent_portfolio(ret)
    spy = st.spy_only(ret)
    p60 = st.sixty_forty(ret)

    gb_s = st.portfolio_stats(gb, rf=rf)
    pp_s = st.portfolio_stats(pp, rf=rf)
    spy_s = st.portfolio_stats(spy, rf=rf)
    p60_s = st.portfolio_stats(p60, rf=rf)

    t_vs_spy = st.hac_tstat_annual(gb, spy)

    print(f"  cycle_strength={cycle_strength:.1f}")
    print(f"    GB (20/20/20/20/20): CAGR={gb_s['cagr']*100:+5.1f}%  Sharpe={gb_s['sharpe']:+.3f}  "
          f"MaxDD={gb_s['max_dd']*100:+5.1f}%")
    print(f"    PP (25/25/25/25):    CAGR={pp_s['cagr']*100:+5.1f}%  Sharpe={pp_s['sharpe']:+.3f}  "
          f"MaxDD={pp_s['max_dd']*100:+5.1f}%")
    print(f"    60/40:               CAGR={p60_s['cagr']*100:+5.1f}%  Sharpe={p60_s['sharpe']:+.3f}  "
          f"MaxDD={p60_s['max_dd']*100:+5.1f}%")
    print(f"    SPY:                 CAGR={spy_s['cagr']*100:+5.1f}%  Sharpe={spy_s['sharpe']:+.3f}  "
          f"MaxDD={spy_s['max_dd']*100:+5.1f}%")
    print(f"    HAC t (GB-SPY ann returns): {t_vs_spy:+.2f}")
    print()


def main() -> None:
    print("Study 203 — Golden-Butterfly — synthetic regime-cycle control\n")
    print("Spine: does the 20/20/20/20/20 blend beat SPY on Sharpe when a regime")
    print("cycle is planted? (positive control) vs when it is absent? (null)\n")
    print("-" * 70)
    for cs in (0.0, 0.3, 0.6):
        _run(cs)

    print("Interpretation:")
    print("  cycle_strength=0  => all legs i.i.d.; GB == equal-weight mix.")
    print("  cycle_strength>0  => planted rotation; the five-asset blend lifts")
    print("  Sharpe and cuts drawdown vs raw equity. On the real tape (2004-2026)")
    print("  the GB adds the small-cap-value leg on top of the Permanent Portfolio.")
    print("  See docs/results.md for real-tape headline numbers.")


if __name__ == "__main__":
    main()
