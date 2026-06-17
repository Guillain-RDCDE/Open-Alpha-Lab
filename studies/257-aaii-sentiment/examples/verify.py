"""Real-tape verification -- Study 257 (AAII-Sentiment). Regenerates docs/results.md numbers.

Joins the curated monthly AAII bull-bear table to the cached ^GSPC month-end
closes, runs the regime sort, the predictive HAC regression, the long/flat timing
overlay vs buy-and-hold, and a sub-period breakdown. Network is touched only with
--fetch (re-pull ^GSPC); the AAII table is always the hardcoded curated snapshot.

    python studies/257-aaii-sentiment/examples/verify.py          # cache-only
    python studies/257-aaii-sentiment/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aaii_sentiment import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.build_real_panel(fetch=fetch)
    print(f"Real tape: {len(panel)} months  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Fingerprint(spread): {data.fingerprint(panel)}\n")

    # ---- Regime sort -------------------------------------------------------
    rs = st.regime_summary(panel)
    print("=== Next-month S&P price return by prior sentiment tercile ===")
    for name in ("low", "mid", "high"):
        row = rs.loc[name]
        print(f"  {name:>4}: {row['mean_ann']*100:+.1f}%/yr  HAC t={row['tstat']:+.2f}  n={int(row['n'])}")
    print(f"  low-minus-high: {(rs.loc['low','mean_ann']-rs.loc['high','mean_ann'])*100:+.1f} pp/yr")

    # ---- Headline long-short + regression ----------------------------------
    contra = st.regime_spread(panel)
    s_contra = st.summarize(contra)
    print("\n=== Contrarian long-short (long panic / short euphoria) ===")
    print(f"  {s_contra['mean']*1200:+.2f}%/yr  HAC t={s_contra['tstat']:+.2f}  n={s_contra['n']}")

    reg = st.predictive_regression(panel)
    print("\n=== Predictive regression: ret_(t+1) ~ standardised prior spread ===")
    print(f"  beta={reg['beta']*100:+.3f} %/sd  HAC t={reg['tstat']:+.2f}  r2={reg['r2']:.4f}  n={reg['n']}")

    # ---- Timing overlay vs buy-and-hold ------------------------------------
    ov = st.timing_overlay(panel, allow_short=False, one_way_bps=5.0)
    s_net = st.summarize(ov["net"]); s_bh = st.summarize(ov["bh"])
    print("\n=== Long/flat contrarian overlay vs buy-and-hold (price-only, 5bps one-way) ===")
    print(f"  Overlay NET: {s_net['mean']*1200:+.2f}%/yr  SR={s_net['sharpe']*np.sqrt(12):+.2f}  t={s_net['tstat']:+.2f}")
    print(f"  Buy & hold : {s_bh['mean']*1200:+.2f}%/yr  SR={s_bh['sharpe']*np.sqrt(12):+.2f}")
    print(f"  -> overlay {'beats' if s_net['mean']>s_bh['mean'] else 'LOSES TO'} buy-and-hold")

    # ---- Sub-periods -------------------------------------------------------
    print("\n=== Sub-period regime breakdown (low vs high) ===")
    for lab, a, b in [("1987-1999", "1987", "1999"), ("2000-2012", "2000", "2012"), ("2013-2026", "2013", "2026")]:
        sub = panel[a:b]; r = st.regime_summary(sub)
        print(f"  {lab}: low={r.loc['low','mean_ann']*100:+.1f}%/yr (t={r.loc['low','tstat']:+.2f})  "
              f"high={r.loc['high','mean_ann']*100:+.1f}%/yr  n={len(sub)}")

    # ---- Synthetic positive control ----------------------------------------
    print("\n=== Synthetic positive control / null ===")
    de, _ = data.synthetic_monthly(edge=0.06, seed=257)
    de = de.assign(ret=de["ret"].shift(-1)).dropna()
    dn, _ = data.synthetic_monthly(edge=0.0, seed=257)
    dn = dn.assign(ret=dn["ret"].shift(-1)).dropna()
    print(f"  edge=0.06: beta t={st.predictive_regression(de)['tstat']:+.2f}  (should be << -2)")
    print(f"  null=0.00: beta t={st.predictive_regression(dn)['tstat']:+.2f}  (should be ~0)")

    print(f"\nFingerprint: {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
