"""Reproduce the real-data headline run (docs/results.md) — oil → equities.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download CL=F + ^GSPC monthly from Yahoo, then run
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

from black_gold import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_pair(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    reg = st.predict_regression(d["oil"], d["eq"])
    print(f"\nOil → equities, {d.index.min().date()}..{d.index.max().date()} ({reg['n']} months)\n")
    print(f"  regress eq_t on oil_(t-1): slope {reg['slope']:+.3f}  r {reg['r']:+.3f}  t={reg['tstat']:+.2f}  "
          "(Driesprong predicts slope < 0; |t| > 2 to be real)")

    timing, bh = st.oil_timing(d["oil"], d["eq"]), st.buy_hold(d["eq"])
    print(f"\n  in market {st.time_in_market(d['oil']):.0%} of the time")
    for nm, s in [("oil timing (long if oil fell)", timing), ("buy & hold S&P", bh)]:
        x = st.summary(s)
        print(f"  {nm:30} CAGR {x['cagr']:+.2%}  Sharpe {x['sharpe']:.2f}  maxDD {x['max_drawdown']:.0%}")

    print("\n  decay (predictive slope / t):")
    for lab, sl in [("2000-2008", d[d.index.year <= 2008]), ("2009-on", d[d.index.year >= 2009])]:
        rr = st.predict_regression(sl["oil"], sl["eq"])
        print(f"    {lab}: slope {rr['slope']:+.3f}  t={rr['tstat']:+.2f}  (n={rr['n']})")

    try:
        from quantlab import repro
        print(f"\nas-of {d.index[-1].date()} · inputs fingerprint {repro.fingerprint(d)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
