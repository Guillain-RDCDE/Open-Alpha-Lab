"""Reproduce the real-data headline run (docs/results.md) — return seasonality on S&P 500 names.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # Wikipedia membership + Yahoo monthly (slow), then run
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

from groundhog import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network; the panel download is slow).")
        return

    same = st.seasonal_hedge(ret, same_month=True)
    ctrl = st.seasonal_hedge(ret, same_month=False)
    print(f"\nReturn-seasonality long-short, {same.index.min().date()}..{same.index.max().date()} "
          f"({len(same)} months, {ret.shape[1]} names)\n")
    s = st.stats(same)
    c = st.stats(ctrl)
    print(f"  same-month seasonality: mean {s['mean_ann']:+.2%}/yr  Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")
    print(f"  control (other months): mean {c['mean_ann']:+.2%}/yr  Sharpe {c['sharpe']:+.2f}  (Lo t={c['tstat']:+.2f})  <- should be ~0 if it's truly seasonal")

    print("\n  decay (same-month Sharpe):")
    for lab, sl in [("1999-2012", same.loc[:"2012"]), ("2013-on", same.loc["2013":])]:
        print(f"    {lab}: {st.stats(sl)['sharpe']:+.2f}")

    be = st.breakeven_cost_bps(same)
    print(f"\n  gross mean {same.mean()*1e4:.0f} bp/mo; break-even ≈ {be:.0f} bp/trade "
          "(the book reshuffles ~fully every month)")
    for c_bps in (5, 10, 20):
        print(f"    net Sharpe at {c_bps:2d} bp/trade: {st.stats(st.net_of_cost(same, c_bps))['sharpe']:+.2f}")

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
