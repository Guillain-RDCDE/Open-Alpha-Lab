"""Reproduce the real-data headline run (docs/results.md) — the size premium across pairs.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download ^RUT/^GSPC + IWM/SPY + IJR/IVV from Yahoo

The sample is pinned to the desk's as-of (quantlab.repro). Leg Sharpes are computed on each pair's
**common months only** (leg_summary aligns by construction), and the pre/post-2010 window split comes
with a circular-block-bootstrap p-value on the difference — on a sample that is entirely
post-publication (Banz 1981, data from 1987), a split can show window-dependence, never decay.
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

from vanishing_act import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_pairs(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    ret = repro.as_of(ret)

    for name, (small, large) in data.PAIRS.items():
        if small not in ret.columns or large not in ret.columns:
            continue
        sp = st.smb(ret, small, large)
        s = st.smb_stats(sp)
        legs = st.leg_summary(ret, small, large)
        note = "price indices, no dividends (a TR spread would be MORE negative)" \
            if small.startswith("^") else "total return"
        print(f"\n{small} − {large}  ({legs['start'].date()}..{legs['end'].date()}, {legs['n']} mo, {note})")
        print(f"  SMB: mean {s['mean_ann']:+.2%}/yr  Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")
        print(f"  legs on the SAME months: small Sharpe {legs['small']['sharpe']:.2f} (CAGR {legs['small']['cagr']:+.2%})"
              f"  vs  large Sharpe {legs['large']['sharpe']:.2f} (CAGR {legs['large']['cagr']:+.2%})")
        d = st.split_difference(sp, cut_year=2010)
        print(f"  window split at 2010 (post-hoc): pre {d['pre_ann']:+.2%}/yr, post {d['post_ann']:+.2%}/yr"
              f"  — diff {d['diff_ann']:+.2%}/yr, block-bootstrap p={d['p_value']:.2f}")

    roll = st.rolling_mean_ann(st.smb(ret, "^RUT", "^GSPC"), 120).dropna()
    print(f"\nTrailing 10-year SMB (^RUT − ^GSPC): ranges {roll.min():+.1%} to {roll.max():+.1%}/yr, "
          f"positive in {float((roll > 0).mean()):.0%} of windows — the sign is a window choice")

    print(f"\n{repro.data_stamp('^RUT/^GSPC + IWM/SPY + IJR/IVV monthly', ret)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
