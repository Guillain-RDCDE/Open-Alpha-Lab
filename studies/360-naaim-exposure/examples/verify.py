"""Real-tape verification -- Study 360 (NAAIM-Exposure). Regenerates docs/results.md.

Joins the weekly NAAIM Exposure Index to SPY total-return closes, runs the regime
sort, the predictive HAC regression, the long/flat contrarian timing overlay vs
buy-and-hold, a sub-period breakdown, and the synthetic positive control / null.
Network is touched only with --fetch (re-pull SPY); NAAIM always comes from the
cached weekly CSV (or the real quarterly fallback if the cache is absent).

    python studies/360-naaim-exposure/examples/verify.py          # cache-only
    python studies/360-naaim-exposure/examples/verify.py --fetch  # refresh SPY
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from naaim_exposure import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.build_real_panel(fetch=fetch)
    print(f"Real tape: {len(panel)} weeks  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"NAAIM range: {panel['naaim'].min():.1f} .. {panel['naaim'].max():.1f}  "
          f"(mean {panel['naaim'].mean():.1f})")
    print(f"Fingerprint(naaim): {data.fingerprint(panel)}\n")

    uncond = st.summarize(panel["ret"])
    print(f"Unconditional weekly SPY return: {uncond['mean']*100:+.3f}%/wk "
          f"({uncond['mean']*5200:+.1f}%/yr)  HAC t={uncond['tstat']:+.2f}  n={uncond['n']}\n")

    # ---- Regime sort -------------------------------------------------------
    rs = st.regime_summary(panel)
    print("=== Next-week SPY total return by prior NAAIM exposure tercile ===")
    for name in ("low", "mid", "high"):
        row = rs.loc[name]
        print(f"  {name:>4}: {row['mean_ann']*100:+6.1f}%/yr  HAC t={row['tstat']:+.2f}  n={int(row['n'])}")
    gap = (rs.loc["low", "mean_ann"] - rs.loc["high", "mean_ann"]) * 100
    print(f"  low-minus-high gap: {gap:+.1f} pp/yr")

    # ---- Headline long-short + regression ----------------------------------
    contra = st.regime_spread(panel)
    sc = st.summarize(contra)
    print("\n=== Contrarian long-short (long cash-regime / short all-in-regime) ===")
    print(f"  {sc['mean']*5200:+.2f}%/yr  HAC t={sc['tstat']:+.2f}  n={sc['n']}")

    reg = st.predictive_regression(panel)
    print("\n=== Predictive regression: ret_(t+1) ~ standardised prior exposure ===")
    print(f"  beta={reg['beta']*100:+.3f} %/sd  HAC t={reg['tstat']:+.2f}  "
          f"r2={reg['r2']:.4f}  n={reg['n']}")

    # ---- Timing overlay vs buy-and-hold ------------------------------------
    ov = st.timing_overlay(panel, allow_short=False, one_way_bps=5.0)
    s_net = st.summarize(ov["net"]); s_bh = st.summarize(ov["bh"])
    print("\n=== Long/flat contrarian overlay vs buy-and-hold (total return, 5bps one-way) ===")
    print(f"  Overlay NET: {s_net['mean']*5200:+.2f}%/yr  SR={s_net['sharpe']*np.sqrt(52):+.2f}  t={s_net['tstat']:+.2f}")
    print(f"  Buy & hold : {s_bh['mean']*5200:+.2f}%/yr  SR={s_bh['sharpe']*np.sqrt(52):+.2f}")
    frac_long = float((ov["pos"] > 0).mean())
    print(f"  overlay in-market {frac_long*100:.0f}% of weeks; "
          f"{'beats' if s_net['mean']>s_bh['mean'] else 'LOSES TO'} buy-and-hold")

    # ---- Sub-periods -------------------------------------------------------
    print("\n=== Sub-period regime breakdown (low vs high) ===")
    for lab, a, b in [("2006-2012", "2006", "2012"),
                      ("2013-2019", "2013", "2019"),
                      ("2020-2026", "2020", "2026")]:
        sub = panel[a:b]; r = st.regime_summary(sub)
        print(f"  {lab}: low={r.loc['low','mean_ann']*100:+6.1f}%/yr (t={r.loc['low','tstat']:+.2f})  "
              f"high={r.loc['high','mean_ann']*100:+6.1f}%/yr (t={r.loc['high','tstat']:+.2f})  n={len(sub)}")

    # ---- Synthetic positive control / null ---------------------------------
    print("\n=== Synthetic positive control / null (machinery proof) ===")
    de, _ = data.synthetic_weekly(edge=0.004, seed=360)
    de = de.assign(ret=de["ret"].shift(-1)).dropna()
    dn, _ = data.synthetic_weekly(edge=0.0, seed=360)
    dn = dn.assign(ret=dn["ret"].shift(-1)).dropna()
    print(f"  edge=0.05: regression beta HAC t={st.predictive_regression(de)['tstat']:+.2f}  (should be << -2)")
    print(f"  null=0.00: regression beta HAC t={st.predictive_regression(dn)['tstat']:+.2f}  (should be ~0)")

    print(f"\nFingerprint(naaim): {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
