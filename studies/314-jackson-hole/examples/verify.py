"""Reproduce the real run (docs/results.md) — the Jackson-Hole event study on SPY.

    python examples/verify.py            # cache-only (reads the shared SPY pull)
    python examples/verify.py --fetch    # refresh the shared SPY cache online, then run
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

from jackson_hole import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_spy(fetch=fetch)
    if ret.empty:
        print("No cached SPY data (run with --fetch once to populate the shared cache).")
        return
    print(f"\nJackson-Hole event study, SPY {ret.index.min().date()}–{ret.index.max().date()} "
          f"({len(ret)} sessions, {len(data.JH_SPEECH_DATES)} keynotes)")
    print(f"inputs fingerprint {data.fingerprint(ret)}\n")

    sw = st.offset_sweep(ret, data.JH_SPEECH_DATES, offsets=range(-5, 6))
    print("  offset  ev_mean(bps)  diff(bps)  welch_t  hac_t  win%")
    for _, r in sw.iterrows():
        print(f"  {int(r['offset']):+6d}  {r['event_mean']*1e4:+11.1f}  {r['diff_bps']:+8.1f}  "
              f"{r['welch_t']:+6.2f}  {r['hac_t']:+5.2f}  {r['win_rate']*100:4.0f}")

    lo, hi = st.block_bootstrap_ci(ret, data.JH_SPEECH_DATES, offset=0)
    print(f"\n  speech-day mean 95% CI (block bootstrap): [{lo*1e4:+.1f}, {hi*1e4:+.1f}] bps")

    print("\n  react-to-the-speech trade (long the next session, 1 bp one-way), net:")
    for h in (1, 2, 3):
        s = st.summarize_trades(st.react_trade(ret, data.JH_SPEECH_DATES, hold=h, cost_bps=1.0), "ret_net")
        print(f"    hold={h}: n={s['n']} win={s['win_rate']*100:.0f}% "
              f"mean={s['mean_bps']:+.1f}bps sharpe={s['sharpe']:+.2f} hac_t={s['hac_t']:+.2f}")

    print("\n  synthetic control (machinery proof):")
    r0, sp0, _ = data.synthetic_world(bump=0.0, seed=314)
    r1, sp1, _ = data.synthetic_world(bump=0.008, seed=314)
    print(f"    null  speech-day HAC t = {st.event_table(r0, sp0, 0)['hac_t']:+.2f}")
    print(f"    +80bps speech-day HAC t = {st.event_table(r1, sp1, 0)['hac_t']:+.2f}")


if __name__ == "__main__":
    main("--fetch" in sys.argv)
