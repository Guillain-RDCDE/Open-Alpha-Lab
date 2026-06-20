"""Reproduce the pinned headline run for Study 331 (Fifty-Two-Week-Range).

Cache-first: reads the 20-name S&P 500 basket from this study's ``_cache`` or the shared
study-202 cache; never touches the network. Prints the fingerprints, the per-horizon
range-position spread, the paired horse-race difference vs the high ratio, the spanning
regression, the long-only comparison, and the cost sweep — the exact numbers mirrored in
``docs/results.md`` and the notebooks' frozen ``R`` dict.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fifty_two_week_range import data, strategy as st  # noqa: E402


def main() -> int:
    if not data.have_real_cache():
        print("No real cache found (this study's _cache or study-202's). "
              "Run once online to populate, or rely on the synthetic control.")
        return 0

    panels = data.load_ohlc_panels(allow_survivorship_bias=True)
    close = panels["close"]
    pos = st.range_position(close)
    hr = st.high_ratio(close)

    print(f"close: {close.index[0].date()} -> {close.index[-1].date()}  "
          f"n={len(close)}  names={close.shape[1]}")
    for f in ("close", "high", "low"):
        print(f"  fingerprint[{f}] = {data.fingerprint(panels[f])}")

    print("\nHorizon | POS spread (bps, t) | HR spread (t) | POS-HR diff (bps, t)")
    for hold in (1, 5, 21, 65):
        fwd = st.forward_returns(close, hold)
        qp = st.quintile_spread(pos, fwd)
        qh = st.quintile_spread(hr, fwd)
        sp = st.summarize(qp["spread_gross"])
        sh = st.summarize(qh["spread_gross"])
        diff = st.diff_tstat(qp["spread_gross"], qh["spread_gross"])
        print(f"  {hold:>3}d  | {sp['mean_bps']:+7.1f}  t={sp['tstat']:+5.2f} | "
              f"t={sh['tstat']:+5.2f} | {diff['mean_diff_bps']:+7.2f}  t={diff['tstat']:+5.2f}")

    print("\nSpanning regression (Fama-MacBeth, z-scored, weekly):")
    for hold in (5, 21):
        fwd = st.forward_returns(close, hold)
        sr = st.spanning_regression(pos, hr, fwd)
        rs = st.spanning_regression(hr, pos, fwd)
        print(f"  {hold:>3}d  incremental POS|HR t={st.hac_tstat(sr['b_target']):+.2f}  "
              f"incremental HR|POS t={st.hac_tstat(rs['b_target']):+.2f}")

    fwd5 = st.forward_returns(close, 5)
    s, b = st.long_top_decile(pos, fwd5, decile_frac=0.20)
    print(f"\nLong-only top quintile (POS, 5d): {st.summarize(s)['ann_pct']:+.1f}%/yr "
          f"vs baseline {st.summarize(b)['ann_pct']:+.1f}%/yr  diff t={st.diff_tstat(s, b)['tstat']:+.2f}")

    print("\nCost sweep on POS Q5-Q1 spread (21d, weekly):")
    fwd21 = st.forward_returns(close, 21)
    for c in (0, 5, 10, 20):
        q = st.quintile_spread(pos, fwd21, cost_bps=c)
        ss = st.summarize(q["spread_net"])
        print(f"  {c:>2} bps: net {ss['mean_bps']:+6.1f} bps  t={ss['tstat']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
