"""Real-tape verification — Study 418 (Money Flow Index).

Reads the cached SPY daily tape (or fetches it once with --fetch), runs the MFI and RSI
as the same long/flat timing rule, races their NET excess-of-cash Sharpe against each
other and against buy-and-hold, sweeps costs, runs the block-permutation placebo and a
block-bootstrap on the MFI-minus-RSI difference, and prints the synthetic positive
control — every number that flows into docs/results.md.  Network is touched only with
--fetch.

    python studies/418-money-flow-index/examples/verify.py            # cache-only
    python studies/418-money-flow-index/examples/verify.py --fetch     # refresh tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from money_flow_index import data, strategy as st  # noqa: E402

ASOF = "2026-06-23"   # the partial 2026-06-24 bar is dropped from the stamped run


def main(fetch: bool) -> None:
    bars = data.fetch_daily("SPY", start="2000-01-01", fetch=fetch).loc[:ASOF]
    print(f"SPY {bars.index[0].date()}..{bars.index[-1].date()} "
          f"n={len(bars)} fp={data.fingerprint(bars)}")
    print()

    res = st.run_experiment(bars, window=14, buy_thr=30.0, sell_thr=70.0,
                            cost_bps=2.0, rf_annual=0.0, perm_iter=2000)

    print("=== net excess-of-cash books (2 bps one-way) ===")
    for arm in ("mfi", "rsi", "buy_hold"):
        s = res[arm]["net"]
        expo = s.get("exposure", 1.0)
        print(f"{arm:8s} sharpe={s['sharpe']:+.3f} ann={s['ann_return']*100:+.2f}% "
              f"HAC_t={s['hac_t']:+.2f} max_dd={s['max_dd']*100:.2f}% expo={expo:.3f}")
    print(f"delta MFI-RSI sharpe = {res['delta_sharpe_mfi_minus_rsi']:+.3f}")
    print(f"delta MFI-buyhold    = {res['delta_sharpe_mfi_minus_bh']:+.3f}")

    print("\n=== block-permutation placebo (timing informative?) ===")
    print(f"MFI perm p = {res['mfi']['perm']['p_value']:.3f}   "
          f"RSI perm p = {res['rsi']['perm']['p_value']:.3f}")

    # --- block-bootstrap on the MFI-minus-RSI daily net-excess difference ---
    pm = st.positions_from_indicator(st.money_flow_index(bars))
    pr = st.positions_from_indicator(st.rsi(bars["close"]))
    dm = (st.backtest(bars, pm, 2.0)["net_excess"]
          - st.backtest(bars, pr, 2.0)["net_excess"]).dropna().to_numpy()
    rng = np.random.default_rng(418)
    n, blk = len(dm), 21
    nb = int(np.ceil(n / blk))
    boot = np.array([np.concatenate(
        [dm[i:i + blk] for i in rng.integers(0, n - blk, size=nb)])[:n].mean()
        for _ in range(2000)])
    pval = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print("\n=== MFI-minus-RSI head-to-head (block bootstrap) ===")
    print(f"observed daily diff mean = {dm.mean():.3e}  two-sided p = {pval:.3f}")

    print("\n=== cost sweep (MFI / RSI net excess Sharpe) ===")
    for c in (0.0, 2.0, 5.0, 10.0):
        r = st.run_experiment(bars, cost_bps=c, perm_iter=1)
        print(f"cost={c:5.1f}  MFI sharpe={r['mfi']['net']['sharpe']:.3f} "
              f"t={r['mfi']['net']['hac_t']:.2f}  "
              f"RSI sharpe={r['rsi']['net']['sharpe']:.3f}")

    print("\n=== synthetic positive control (seed 418, net excess Sharpe) ===")
    for edge in (0.0, 1.0, 2.0):
        b, _ = data.synthetic_panel(n_days=4000, edge=edge, seed=418)
        r = st.run_experiment(b, cost_bps=2.0, perm_iter=1)
        m, rr = r["mfi"]["net"]["sharpe"], r["rsi"]["net"]["sharpe"]
        print(f"edge={edge:4.1f}  MFI {m:+.3f}  RSI {rr:+.3f}  delta {m - rr:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
