"""Reproduce the real-data headline run (docs/results.md) — value vs growth across pairs.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download IVE/IVW + VTV/VUG + RPV/RPG from Yahoo
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

from bargain_bin import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_pairs(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    for name, (val, grw) in data.PAIRS.items():
        if val not in ret.columns or grw not in ret.columns:
            continue
        sp = st.hml(ret, val, grw)
        s = st.hml_stats(sp)
        lv, lg = st.leg_summary(ret, val), st.leg_summary(ret, grw)
        print(f"\n{val} − {grw}  ({sp.index.min().date()}..{sp.index.max().date()}, {s['n']} mo)")
        print(f"  HML: mean {s['mean_ann']:+.2%}/yr  Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")
        print(f"  value Sharpe {lv['sharpe']:.2f} (CAGR {lv['cagr']:+.2%}) vs growth Sharpe {lg['sharpe']:.2f} (CAGR {lg['cagr']:+.2%})")
        print("  regimes:\n" + st.regime_split(sp).round(3).to_string().replace("\n", "\n  "))

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
