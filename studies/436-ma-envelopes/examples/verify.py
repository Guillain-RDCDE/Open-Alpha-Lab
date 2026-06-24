"""Real-tape verification — Study 436 (MA Envelopes). Regenerates docs/results.md numbers.

Reads the cached daily tapes (or fetches with --fetch), runs the SMA(20) ±5% percent
envelope as a long/flat timing rule against Bollinger, SMA(200) and buy-and-hold — net of
one-way costs, excess-of-cash, with one execution lag — plus the block-permutation placebo,
cost sweep, panel breakdown, breakout reading and the synthetic positive control. Network is
touched only with --fetch.

    python studies/436-ma-envelopes/examples/verify.py            # cache-only
    python studies/436-ma-envelopes/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ma_envelopes import data, strategy as st  # noqa: E402

ASOF = "2026-06-23"
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "EFA"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _close(t: str, fetch: bool) -> pd.Series:
    return data.fetch_daily(t, fetch=fetch, cache_dir=CACHE).loc[:ASOF]["close"]


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE).loc[:ASOF]
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    c = _close("SPY", False)

    print("=== HEADLINE race — SPY, net of 1 bp one-way, excess-of-cash ===")
    res = st.run_experiment(c, n=20, k=0.05, mode="reversion", cost_bps=1.0, n_perm=2000)
    for name in ["envelope", "bollinger", "sma", "buyhold"]:
        s = res[name]
        print(f"{name:10s} Sharpe={s['sharpe']:+.3f} t={s['tstat']:+.2f} "
              f"cagr={s['cagr']:+.4f} maxdd={s['maxdd']:+.3f} exp={s['exposure']:.3f}")

    print("\n=== Block-permutation placebo (scramble timing, keep exposure) ===")
    p = res["perm"]
    print(f"real Sharpe={p['stat_real']:+.3f} perm_mean={p['perm_mean']:+.3f} "
          f"p={p['p_value']:.3f}")

    print("\n=== Cost sweep (SPY envelope reversion) ===")
    for cost in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(st.book_returns(c, st.envelope_position(c, 20, 0.05, "reversion"), cost), "net")
        print(f"cost={cost:.1f}bp  Sharpe={s['sharpe']:+.3f} t={s['tstat']:+.2f} cagr={s['cagr']:+.4f}")

    print("\n=== Panel (envelope vs Bollinger vs buy-and-hold, net 1 bp) ===")
    for t in TICKERS:
        cc = _close(t, False)
        e = st.summarize(st.book_returns(cc, st.envelope_position(cc, 20, 0.05, "reversion"), 1.0), "net")
        bo = st.summarize(st.book_returns(cc, st.bollinger_position(cc, 20, 2.0, "reversion"), 1.0), "net")
        bh = st.summarize(pd.DataFrame(
            {"net": st.buy_and_hold_excess(cc), "pos": pd.Series(1.0, index=cc.index)}), "net")
        print(f"{t:5s} env S={e['sharpe']:+.3f}(t={e['tstat']:+.2f}) "
              f"boll S={bo['sharpe']:+.3f}(t={bo['tstat']:+.2f}) bh S={bh['sharpe']:+.3f}")

    print("\n=== Breakout reading (SPY) ===")
    for nm, pos in [("env-breakout", st.envelope_position(c, 20, 0.05, "breakout")),
                    ("boll-breakout", st.bollinger_position(c, 20, 2.0, "breakout"))]:
        s = st.summarize(st.book_returns(c, pos, 1.0), "net")
        print(f"{nm:14s} Sharpe={s['sharpe']:+.3f} t={s['tstat']:+.2f}")

    print("\n=== Synthetic positive control (planted mean-reversion) ===")
    for edge in (0.0, 0.02, 0.05, 0.10):
        b, _ = data.synthetic_panel(n_days=4000, edge=edge, seed=436)
        cc = b["close"]
        pos = st.envelope_position(cc, 20, 0.05, "reversion")
        s = st.summarize(st.book_returns(cc, pos, 1.0), "net")
        pp = st.block_permutation_pvalue(cc, pos, 1.0, n_perm=800, seed=7)
        print(f"edge={edge:.2f}  Sharpe={s['sharpe']:+.3f} exp={s['exposure']:.3f} "
              f"placebo_p={pp['p_value']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
