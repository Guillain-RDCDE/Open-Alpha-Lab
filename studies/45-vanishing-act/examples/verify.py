"""Reproduce the real-data headline run (docs/results.md) — the size premium across pairs.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download ^RUT/^GSPC + IWM/SPY + IJR/IVV from Yahoo
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


def main(fetch: bool) -> None:
    ret = data.fetch_pairs(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    for name, (small, large) in data.PAIRS.items():
        if small not in ret.columns or large not in ret.columns:
            continue
        sp = st.smb(ret, small, large)
        s = st.smb_stats(sp)
        ls, ll = st.leg_summary(ret, small), st.leg_summary(ret, large)
        print(f"\n{small} − {large}  ({sp.index.min().date()}..{sp.index.max().date()}, {s['n']} mo)")
        print(f"  SMB: mean {s['mean_ann']:+.2%}/yr  Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")
        print(f"  small Sharpe {ls['sharpe']:.2f} (CAGR {ls['cagr']:+.2%}) vs large Sharpe {ll['sharpe']:.2f} (CAGR {ll['cagr']:+.2%})")
        print("  decay:\n" + st.decay_split(sp).round(3).to_string().replace("\n", "\n  "))

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
