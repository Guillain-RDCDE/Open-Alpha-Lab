"""Reproduce the real-data headline run (docs/results.md).

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download ^GSPC (effect) + SPY (tradable) + ^IRX from Yahoo

The effect on the S&P 500 price index back to 1950; the tradable window book on SPY with its cash leg
credited at the T-bill, raced against buy-and-hold in both Sharpe conventions; the decay with a
formal sub-period test rather than a bare assertion.
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

from last_call import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    gspc = data.fetch_prices("^GSPC", fetch=fetch, start_year=1950)   # price-only index, long history
    spy = data.fetch_prices("SPY", fetch=fetch, start_year=1993)      # total return (auto-adjusted)
    if gspc.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    rf = data.fetch_tbill(fetch=fetch).reindex(spy.index).ffill().fillna(0.0)

    d = st.tom_vs_rest(gspc)
    print(f"\nS&P 500 (^GSPC, price-only) {gspc.index.min().date()}..{gspc.index.max().date()} — "
          f"{d['n_tom'] + d['n_rest']} days ({d['n_tom']} TOM + {d['n_rest']} rest)")
    print(f"  TOM days ({d['tom_share']:.0%} of days): {d['tom_bp']:.1f} bp/day  |  non-TOM: {d['rest_bp']:.1f} bp/day  |  Welch t={d['welch_t']:.1f}")

    print(f"\nTradable on SPY (total return) {spy.index.min().date()}..{spy.index.max().date()} — "
          f"cash leg credited at the ^IRX T-bill; Sharpe quoted raw and excess-of-cash:")
    for c in (0.0, 2.0, 5.0):
        book = st.tom_returns(spy, cost_bps=c, rf=rf)
        s, sx = st.summary(book), st.summary(book, rf=rf)
        print(f"  TOM-only (cost {c:.0f}bp): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f} (excess {sx['sharpe']:.2f})  "
              f"maxDD {s['max_drawdown']:.0%}  in-mkt {st.tom_mask(spy.index).mean():.0%}")
    bh = st.buy_hold(spy)
    s, sx = st.summary(bh), st.summary(bh, rf=rf)
    print(f"  Buy & hold SPY:        CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f} (excess {sx['sharpe']:.2f})  "
          f"maxDD {s['max_drawdown']:.0%}  in-mkt 100%")

    print("\nDecay (TOM vs non-TOM bp/day, each slice's own Welch t):")
    for lab, lo, hi in [("1950-1987", 1950, 1987), ("1988-2007", 1988, 2007), ("2008-on", 2008, 2100)]:
        sl = gspc[(gspc.index.year >= lo) & (gspc.index.year <= hi)]
        dd = st.tom_vs_rest(sl)
        print(f"  {lab}: TOM {dd['tom_bp']:5.1f}  non-TOM {dd['rest_bp']:5.1f}  (t={dd['welch_t']:.2f})")
    pc = st.tom_premium_change(gspc, split="2008")
    print(f"  Formal test of the fade: premium {pc['premium_pre_bp']:.1f} bp (pre-2008, t={pc['welch_t_pre']:.1f}) → "
          f"{pc['premium_post_bp']:.1f} bp (2008-on, t={pc['welch_t_post']:.1f}); change t={pc['t_change']:.2f}")

    try:
        from quantlab import repro
        print(f"\nas-of {gspc.index[-1].date()} · ^GSPC fingerprint {repro.fingerprint(gspc.to_frame())} · "
              f"SPY fingerprint {repro.fingerprint(spy.to_frame())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
