"""Reproduce the real-data headline run (docs/results.md) — the 52-week-high effect on S&P 500 names.

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

from high_water import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    # Opt-in stated: current S&P 500 membership + >=20y-history filter = survivor panel. For this
    # study the bias cuts at the *sign*: the short leg (fallen names) is guaranteed survivors, so
    # the negative hedge is partly manufactured; the momentum correlation is the bias-robust half.
    ret = data.fetch_panel(fetch=fetch, allow_survivorship_bias=True)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network; the panel download is slow).")
        return

    hi = st.cross_section_hedge(ret, st.nearness(ret))
    mo = st.cross_section_hedge(ret, st.momentum(ret))
    print(f"\n52-week-high vs momentum, {hi.index.min().date()}..{hi.index.max().date()} "
          f"({len(hi)} months, {ret.shape[1]} names; survivor panel — the sign is partly its artifact)\n")
    h, m = st.stats(hi), st.stats(mo)
    print(f"  52-week-high nearness: mean {h['mean_ann']:+.2%}/yr  Sharpe {h['sharpe']:+.2f}  (Lo t={h['tstat']:+.2f})  hit {h['hit_rate']:.0%}")
    print(f"  12-2 momentum:         mean {m['mean_ann']:+.2%}/yr  Sharpe {m['sharpe']:+.2f}  (Lo t={m['tstat']:+.2f})")
    print(f"  corr(52w-high hedge, 12-2 momentum hedge) = {st.signal_overlap(ret):+.2f}  <- ~1 ⇒ same factor")

    print("\n  decay (52w-high Sharpe):")
    for lab, sl in [("1996-2012", hi.loc[:"2012"]), ("2013-on", hi.loc["2013":])]:
        print(f"    {lab}: {st.stats(sl)['sharpe']:+.2f}")
    print("  net Sharpe after one-way costs (3.2x NAV traded/mo):")
    for c in (5, 10, 20):
        print(f"    {c:2d} bp one-way: {st.stats(st.net_of_cost(hi, c))['sharpe']:+.2f}")

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
