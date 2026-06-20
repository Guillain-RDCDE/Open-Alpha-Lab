"""Real- & synthetic-tape verification — Study 344 (Backtest-Overfitting). Regenerates
docs/results.md numbers.

Runs the grid-search overfitting demonstration on three tapes — the synthetic null (no
edge), the synthetic positive control (a real but un-timeable drift), and real SPY — and
reports the in-sample champion's IS vs OOS Sharpe, the Deflated Sharpe Ratio, and the PBO
(CSCV).

    python studies/344-backtest-overfitting/examples/verify.py           # cache-only
    python studies/344-backtest-overfitting/examples/verify.py --fetch    # refresh SPY
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backtest_overfitting import data, strategy as st  # noqa: E402

ND = 1260
N_TRIALS = 1000
N_SPLITS = 10


def _report(name: str, mat: pd.DataFrame) -> None:
    champ = st.in_sample_champion(mat, frac=0.5)
    dsr = st.deflated_sharpe_ratio(mat[champ["champion"]], n_trials=mat.shape[1])
    pbo = st.pbo_cscv(mat, n_splits=N_SPLITS)
    t = st.hac_tstat(champ["oos_returns"])
    print(f"=== {name} (N={mat.shape[1]} trials) ===")
    print(f"  champion           : {champ['champion']}")
    print(f"  IS Sharpe          : {champ['is_sharpe']:+.2f}")
    print(f"  OOS Sharpe         : {champ['oos_sharpe']:+.2f}")
    print(f"  OOS HAC t          : {t['tstat']:+.2f}  (n={t['n']})")
    print(f"  expected-max SR0   : {dsr['sr0']:.2f}")
    print(f"  Deflated Sharpe    : {dsr['dsr']:.3f}")
    print(f"  PBO (CSCV)         : {pbo['pbo']:.3f}  ({pbo['n_combos']} partitions)\n")


def main(fetch: bool) -> None:
    # --- synthetic null -----------------------------------------------------
    prices_null, _ = data.synthetic_prices(n_days=ND, edge=0.0, seed=344)
    print(f"synthetic null fingerprint   : {data.fingerprint(prices_null)}")
    mat_null = st.sweep(prices_null, n_trials=N_TRIALS, cost_bps=0.0)
    _report("SYNTHETIC NULL (edge=0)", mat_null)

    # --- synthetic positive control ----------------------------------------
    prices_edge, _ = data.synthetic_prices(n_days=ND, edge=0.0003, seed=344)
    mat_edge = st.sweep(prices_edge, n_trials=N_TRIALS, cost_bps=0.0)
    bh_edge = st.buy_and_hold(prices_edge)
    dbh = st.deflated_sharpe_ratio(bh_edge, n_trials=1)
    print(f"control buy-hold Sharpe      : {st.sharpe(bh_edge):.2f}  "
          f"DSR(N=1)={dbh['dsr']:.3f}  <- the honest single hypothesis survives")
    _report("SYNTHETIC CONTROL (edge=0.0003, un-timeable drift)", mat_edge)

    # --- real SPY -----------------------------------------------------------
    try:
        px = data.load_real("SPY", fetch=fetch)
        today = pd.Timestamp.today().normalize()
        px = px[px.index < today.to_period("M").to_timestamp()]  # drop partial month
        print(f"SPY {px.index[0].date()} .. {px.index[-1].date()}  "
              f"({len(px)} days)  fingerprint: {data.fingerprint(px)}")
        mat_spy = st.sweep(px, n_trials=N_TRIALS, cost_bps=0.0)
        bh = st.buy_and_hold(px)
        print(f"SPY buy-hold Sharpe          : {st.sharpe(bh):.2f}  "
              f"DSR(N=1)={st.deflated_sharpe_ratio(bh, n_trials=1)['dsr']:.3f}")
        _report("REAL SPY 2000-2026", mat_spy)
    except FileNotFoundError as e:
        print(f"\n[SPY skipped — no cache; run with --fetch to populate] {e}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
