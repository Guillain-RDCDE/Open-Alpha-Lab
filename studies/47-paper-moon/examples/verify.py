"""Reproduce the real-data headline run (docs/results.md) — the Fed Model on Shiller data.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download Shiller's S&P series (datahub mirror), then run
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

from paper_moon import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_shiller(fetch=fetch, start_year=1900)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    sig = st.fed_signal(d["ep"], d["y10"])
    timing = st.fed_timing(d["tr"], d["bond"], sig)
    bh = st.buy_hold(d["tr"]).reindex(timing.index)
    print(f"\nFed Model timing, {timing.index.min().date()}..{timing.index.max().date()} ({len(timing)} mo, "
          f"in stocks {st.time_in_stocks(sig.reindex(d.index)):.0%} of the time)\n")
    for nm, s in [("Fed-model timing", timing), ("Buy & hold S&P", bh)]:
        x = st.summary(s)
        print(f"  {nm:22} CAGR {x['cagr']:+.2%}  Sharpe {x['sharpe']:.2f}  maxDD {x['max_drawdown']:.0%}")

    print("\nDoes the bond-yield term add anything? (predictive corr with next-12m equity return)")
    print(f"  Fed signal (E/P − y10): {st.predictive_corr(sig, d['tr']):+.2f}")
    print(f"  E/P alone:              {st.predictive_corr(d['ep'], d['tr']):+.2f}   <- if ≥ the above, the bond term is inert")

    print("\nAsness money-illusion check:")
    print(f"  corr(E/P, 10y yield)        = {d['ep'].corr(d['y10']):+.2f}  (the model assumes they track)")
    print(f"  corr(E/P, trailing inflation) = {d['ep'].corr(d['infl']):+.2f}  (what E/P actually tracks)")

    try:
        from quantlab import repro
        print(f"\nas-of {d.index[-1].date()} · inputs fingerprint {repro.fingerprint(d[['tr','ep','y10']].dropna())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
