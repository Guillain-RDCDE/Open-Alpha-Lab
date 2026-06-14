"""Offline, deterministic demo — Study 144 (Permanent-Portfolio).

No network. Builds a synthetic four-asset world and shows the study's spine in
one screen: the PP improves risk-adjusted return (Sharpe) over 100% stocks *when*
a regime cycle is planted — but the gain dissolves on a no-cycle i.i.d. world.
The regime knob is the only forecastable structure in the tape.

Run:
    python studies/144-permanent-portfolio/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from permanent_portfolio import data, strategy as st  # noqa: E402

SYNTH_MAP = {"STK": "SPY", "BOND": "TLT", "GOLD": "GLD", "CASH": "SHY"}


def _run(cycle_strength: float) -> None:
    frame, _ = data.synthetic_four_asset(n_years=20, cycle_strength=cycle_strength, seed=144)
    ret = st.to_returns(frame).rename(columns=SYNTH_MAP)
    rf = ret["SHY"]

    pp = st.permanent_portfolio(ret)
    spy = st.spy_only(ret)
    p60 = st.sixty_forty(ret)

    pp_s = st.portfolio_stats(pp, rf=rf)
    spy_s = st.portfolio_stats(spy, rf=rf)
    p60_s = st.portfolio_stats(p60, rf=rf)

    t_vs_spy = st.hac_tstat_annual(pp, spy)

    print(f"  cycle_strength={cycle_strength:.1f}")
    print(f"    PP:    CAGR={pp_s['cagr']*100:+5.1f}%  Sharpe={pp_s['sharpe']:+.3f}  "
          f"MaxDD={pp_s['max_dd']*100:+5.1f}%")
    print(f"    SPY:   CAGR={spy_s['cagr']*100:+5.1f}%  Sharpe={spy_s['sharpe']:+.3f}  "
          f"MaxDD={spy_s['max_dd']*100:+5.1f}%")
    print(f"    60/40: CAGR={p60_s['cagr']*100:+5.1f}%  Sharpe={p60_s['sharpe']:+.3f}  "
          f"MaxDD={p60_s['max_dd']*100:+5.1f}%")
    print(f"    HAC t (PP-SPY ann returns): {t_vs_spy:+.2f}")
    print()


def main() -> None:
    print("Study 144 — Permanent-Portfolio — synthetic regime-cycle control\n")
    print("Spine: does the 25/25/25/25 blend beat SPY on Sharpe when a regime")
    print("cycle is planted? (positive control) vs when it is absent? (null)\n")
    print("-" * 65)
    for cs in (0.0, 0.3, 0.6):
        _run(cs)

    print("Interpretation:")
    print("  cycle_strength=0 => all legs i.i.d.; PP ~= equal-weight mix.")
    print("  cycle_strength>0 => planted rotation; PP's cross-asset hedge")
    print("  lifts Sharpe and cuts drawdown vs raw equity. The engine works")
    print("  when the structure is there. On the real tape see docs/results.md.")


if __name__ == "__main__":
    main()
