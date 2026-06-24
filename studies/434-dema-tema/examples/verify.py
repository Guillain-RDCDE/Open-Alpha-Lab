"""Real-tape verification — Study 434 (DEMA & TEMA).

Reads the cached daily SPY tape (cache-first; network only on an explicit cache miss),
races the long/flat trend rule across SMA / EMA / DEMA / TEMA at the 50-day window, runs
the block-permutation placebo and the cost sweep, and reports the honest numbers that flow
into docs/results.md.

    python studies/434-dema-tema/examples/verify.py            # cache-first
    python studies/434-dema-tema/examples/verify.py --fetch    # allow a network refresh

The stamped run drops the partial June 2026 month (as-of 2026-05-31).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dema_tema import data, strategy as st  # noqa: E402

ASOF = "2026-05-31"
WINDOW = 50
COST = 2.0


def main(fetch: bool) -> None:
    b = data.load_real("SPY", fetch=fetch)
    b = b[b.index <= ASOF]                      # drop the partial month
    print(f"SPY {b.index[0].date()}..{b.index[-1].date()} "
          f"n={len(b)} fp={data.fingerprint(b)}  as-of {ASOF}")
    print()

    res = st.run_experiment(b, n=WINDOW, cost_bps=COST, n_draws=5000,
                            placebo_for="tema", seed=434)
    tab = res["table"]
    print(f"=== Sharpe race (n={WINDOW}, one-way cost {COST} bps, long/flat, 1-day lag) ===")
    print(f"buy-and-hold net excess Sharpe: {res['sharpe_bh']:+.3f}")
    print(tab[["sharpe_gross", "sharpe_net", "sharpe_minus_bh", "hac_t_vs_bh",
               "turnover_per_yr", "time_in_market", "cagr_net"]].round(4).to_string())
    print()

    print("=== block-permutation placebo (same activity level, random timing) ===")
    for k in ("dema", "tema"):
        plc = st.block_permutation_pvalue(b, k, WINDOW, cost_bps=COST,
                                          n_draws=5000, seed=434)
        print(f"{k.upper():5s} obs Sharpe={plc['obs_sharpe']:+.3f} "
              f"placebo mean={plc['placebo_mean']:+.3f} p={plc['p_value']:.3f}")
    print()

    print("=== paired daily X-minus-SMA net return (HAC t) ===")
    sma_net = st.backtest(b, "sma", WINDOW, COST)["strat_net"]
    for k in ("dema", "tema"):
        d = st.backtest(b, k, WINDOW, COST)["strat_net"] - sma_net
        print(f"{k.upper():5s}-minus-SMA  HAC t={st.hac_t(d):+.2f}  "
              f"ann mean={d.mean() * 252 * 100:+.2f}%")
    print()

    print("=== cost sweep, net Sharpe (n=50) ===")
    print("cost_bps   SMA     DEMA    TEMA")
    for c in (0.0, 2.0, 5.0, 10.0):
        sm = st.backtest(b, "sma", WINDOW, c)["sharpe_net"]
        dm = st.backtest(b, "dema", WINDOW, c)["sharpe_net"]
        tm = st.backtest(b, "tema", WINDOW, c)["sharpe_net"]
        print(f"{c:6.1f}   {sm:+.3f}  {dm:+.3f}  {tm:+.3f}")
    print()

    print("=== synthetic positive control (n=50, 2bps): SMA harvests a planted trend, TEMA whipsaws ===")
    for e in (0.0, 1.0, 2.0):
        bs, _ = data.synthetic_panel(n_days=4000, edge=e, seed=434)
        r = st.run_experiment(bs, n=WINDOW, cost_bps=COST, n_draws=1)
        print(f"edge={e:.1f}  bh={r['sharpe_bh']:+.3f}  "
              f"sma={r['table'].loc['sma','sharpe_net']:+.3f}  "
              f"tema={r['table'].loc['tema','sharpe_net']:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
