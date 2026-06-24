"""Real-tape verification — Study 419 (Chaikin Money Flow). Regenerates docs/results.md.

Builds CMF(20) on SPY (and a small panel), turns it into a long/flat timing rule, races
it NET (excess-of-cash) against an SMA(50/200) filter, a MACD filter and buy-and-hold,
runs a block-permutation placebo on the CMF timing, sweeps costs, and checks the
synthetic positive control. Network is touched only with --fetch (cache-first otherwise).

    python studies/419-chaikin-money-flow/examples/verify.py            # cache-only
    python studies/419-chaikin-money-flow/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chaikin_money_flow import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = "2026-06-23"
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE", "GLD"]


def _load(t: str, fetch: bool):
    b = data.fetch_daily(t, fetch=fetch, cache_dir=CACHE)
    return b[b.index <= ASOF]


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    b = _load("SPY", fetch)
    print(f"SPY headline tape: {b.index[0].date()}..{b.index[-1].date()} "
          f"n={len(b)} fp={data.fingerprint(b)}")

    res = st.run_experiment(b, window=20, cmf_thr=0.0, cost_bps=2.0, perm_iter=2000)

    print("\n=== The benchmark race (net excess, 2 bps one-way) ===")
    for arm, label in [("cmf", "CMF long/flat"), ("sma", "SMA 50/200"),
                       ("macd", "MACD"), ("buy_hold", "Buy & hold")]:
        s = res[arm]["net"]
        print(f"{label:14s} sharpe={s['sharpe']:+.3f} hac_t={s['hac_t']:+.3f} "
              f"ann={s['ann_return']:+.4f} dd={s['max_dd']:+.3f}")
    print(f"CMF gross: sharpe={res['cmf']['gross']['sharpe']:+.3f} "
          f"hac_t={res['cmf']['gross']['hac_t']:+.3f}")
    print(f"Delta CMF - SMA  : {res['delta_sharpe_cmf_minus_sma']:+.3f}")
    print(f"Delta CMF - MACD : {res['delta_sharpe_cmf_minus_macd']:+.3f}")
    print(f"Delta CMF - B&H  : {res['delta_sharpe_cmf_minus_bh']:+.3f}")

    print("\n=== Permutation placebo (CMF timing) ===")
    p = res["cmf"]["perm"]
    print(f"observed={p['observed_mean']:+.6f} placebo={p['placebo_mean']:+.6f} "
          f"p_value={p['p_value']:.3f}")

    print("\n=== Cost sweep (SPY, CMF net excess) ===")
    cmf = st.chaikin_money_flow(b, 20)
    pos = st.positions_from_cmf(cmf)
    for c in (0.0, 1.0, 2.0, 5.0):
        s = st.summarize_book(st.backtest(b, pos, cost_bps=c)["net_excess"])
        print(f"cost={c:.1f}bps  sharpe={s['sharpe']:+.3f}  hac_t={s['hac_t']:+.3f}")

    print("\n=== Panel (CMF net excess, 2 bps) vs each name's buy-and-hold ===")
    for t in TICKERS:
        if not os.path.exists(data._cache_path(t, CACHE)):
            continue
        bb = _load(t, fetch)
        cb = st.summarize_book(st.backtest(bb, st.positions_from_cmf(
            st.chaikin_money_flow(bb, 20)), cost_bps=2.0)["net_excess"])
        bh = st.summarize_book(st.buy_hold_excess(bb))
        print(f"{t:5s} cmf_sharpe={cb['sharpe']:+.3f} hac_t={cb['hac_t']:+.3f} "
              f"| bh_sharpe={bh['sharpe']:+.3f} n={cb['n_days']}")

    print("\n=== Synthetic positive control (edge sweep; CMF Sharpe, delta vs B&H) ===")
    for e in (0.0, 0.5, 1.0, 2.0):
        sb, _ = data.synthetic_panel(n_days=4000, edge=e, seed=419)
        cs = st.summarize_book(st.backtest(sb, st.positions_from_cmf(
            st.chaikin_money_flow(sb, 20)), cost_bps=0.0)["net_excess"])["sharpe"]
        bhs = st.summarize_book(st.buy_hold_excess(sb))["sharpe"]
        print(f"edge={e:.1f} cmf_sharpe={cs:+.3f} delta_vs_bh={cs - bhs:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
