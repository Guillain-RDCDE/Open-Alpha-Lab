"""Reproduce the real-data headline run (docs/results.md) — the idio-vol puzzle on S&P 500 names.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # Wikipedia membership + Yahoo daily (slow), then run
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

from static import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    daily, mkt = data.fetch_panel(fetch=fetch)
    if daily.empty:
        print("No cached real data. Re-run with --fetch (needs network; the daily panel is large).")
        return

    sig = st.idiovol_signal(daily, mkt)
    textbook = st.cross_section_hedge(daily, sig, long_high=False)   # long low idio-vol
    s = st.stats(textbook)
    print(f"\nIdiosyncratic-volatility puzzle, {textbook.index.min().date()}..{textbook.index.max().date()} "
          f"({len(textbook)} months, {daily.shape[1]} names)\n")
    print(f"  textbook trade (long low-idiovol / short high-idiovol): mean {s['mean_ann']:+.2%}/yr  "
          f"Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")
    print("  >0 ⇒ the puzzle (low-vol wins); <0 ⇒ high-idiovol names beat low (inverted)")

    print("\n  decay (textbook Sharpe):")
    for lab, sl in [("2000-2012", textbook.loc[:"2012"]), ("2013-on", textbook.loc["2013":])]:
        print(f"    {lab}: {st.stats(sl)['sharpe']:+.2f}")

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(daily.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
