"""Real-tape verification — Study 118 (Fed-Model). Regenerates docs/results.md numbers.

Reads the shared Shiller cache, computes the Fed-Model spread (E/P minus 10y nominal
yield) and the Asness (2003) corrected spread (E/P minus real rate), runs in-sample
OLS with Newey–West HAC standard errors, computes OOS R² (train 1871–1969, test
1970–2023), and evaluates the 1-year market-timing strategy.

    python studies/118-fed-model/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fed_model import data, strategy as st  # noqa: E402


def main() -> None:
    panel = data.load_shiller()
    fp = data.fingerprint(panel)
    end = panel[panel["EP"] > 0].index.max()
    start = panel[panel["EP"] > 0].index.min()
    print(f"Shiller data: {start.date()} to {end.date()}  fingerprint={fp}")
    print(f"Total rows with valid EP+rate: {panel[['EP','rate']].dropna().shape[0]}")
    print()

    # ---- In-sample regressions ----
    print("=== In-sample OLS with HAC SE (lags=120 for 10yr overlap) ===")
    for sig, hor, col in [
        ("fed_spread", 10, "fwd10"),
        ("fed_spread", 1,  "fwd1"),
        ("EP",         10, "fwd10"),
        ("ep_vs_real", 10, "fwd10"),
        ("ep_vs_real", 1,  "fwd1"),
    ]:
        sub = panel.dropna(subset=[sig, col])
        lags = 120 if hor == 10 else 12
        res = st.ols_hac(sub[sig].values, sub[col].values, lags=lags)
        print(
            f"{sig} -> {hor}yr:  slope={res['slope']:+.3f}  R2={res['r2']:.3f}  "
            f"HAC_t={res['tstat']:+.2f}  n={res['n']}"
        )

    print()
    print("=== Out-of-sample R² (train 1871–1969, test 1970–2023) ===")
    for sig, col in [
        ("fed_spread", "fwd10"),
        ("EP",         "fwd10"),
        ("ep_vs_real", "fwd10"),
        ("fed_spread", "fwd1"),
    ]:
        oos = st.oos_r2(panel, sig, col, train_end_year=1969)
        print(
            f"{sig} -> {col}:  OOS_R2={oos['oos_r2']:+.3f}  "
            f"n_train={oos['n_train']}  n_test={oos['n_test']}"
        )

    print()
    print("=== 1-year market-timing (signal > expanding median -> long) ===")
    sub1 = panel.dropna(subset=["fed_spread", "fwd1"])
    timing = st.timing_returns(sub1, "fed_spread", "fwd1")
    summ = st.summarize(timing)
    print(
        f"Strategy:   mean {summ['strat_mean_pa']:+.3f}/yr  "
        f"Sharpe {summ['strat_sharpe']:+.3f}  t={summ['strat_tstat']:+.2f}"
    )
    print(
        f"Buy & hold: mean {summ['bh_mean_pa']:+.3f}/yr  "
        f"Sharpe {summ['bh_sharpe']:+.3f}"
    )
    print(
        f"Active:     mean {summ['active_mean_pa']:+.3f}/yr  "
        f"t={summ['active_tstat']:+.2f}"
    )


if __name__ == "__main__":
    main()
