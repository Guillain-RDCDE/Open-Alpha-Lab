"""Reproducible headline run for Study 376 — MOC-Imbalance.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached OHLC under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moc_imbalance import data, strategy as st

print("# MOC-Imbalance — closing-pressure displacement proxy & overnight reversal")
if data.have_real():
    panel = data.load_real()
    span_years = (panel["date"].max() - panel["date"].min()).days / 365.25
    print(f"panel rows     : {len(panel):,}  ({panel['date'].min().date()} -> "
          f"{panel['date'].max().date()}, {span_years:.1f} years)")
    print(f"rebalance days : {int(panel['rebal'].sum())}  (third Friday of quarter-end)")
    print(f"proxy          : disp = (close-open)/(high-low) in [-1,1]  (PROXY for MOC imbalance)")

    print("\n# Reversal regression  overnight ~ a + b*disp  (HAC t, placebo p)")
    print(f"  {'sample':>18} {'n':>7} {'b':>10} {'HAC_t':>8} {'p_plac':>8} "
          f"{'corr':>8} {'fade_win':>9}")
    for name, p in (("All days", panel),
                    ("Rebalance days", st.subset(panel, rebal_only=True))):
        s = st.summarize(p)
        print(f"  {name:>18} {s['n']:>7,} {s['b']:>10.5f} {s['t']:>8.2f} "
              f"{s['p_placebo']:>8.3f} {s['corr']:>8.3f} {s['fade_win']*100:>8.1f}%")
    for tk in ("SPY", "QQQ", "IWM"):
        p = panel[panel["ticker"] == tk]
        s = st.summarize(p)
        print(f"  {tk:>18} {s['n']:>7,} {s['b']:>10.5f} {s['t']:>8.2f} "
              f"{s['p_placebo']:>8.3f} {s['corr']:>8.3f} {s['fade_win']*100:>8.1f}%")

    print("\n# Fade strategy net of a round-trip cost (close in, open out)")
    fr = st.fade_returns(panel)
    sharpe = fr.mean() / fr.std() * np.sqrt(252)
    print(f"  gross daily Sharpe : {sharpe:.3f}")
    for cb in (0.0, 1.0, 2.0):
        c = st.net_of_costs(panel, cost_bps=cb)
        print(f"  cost={cb:>3.0f}bps  net/trade={c['net_mean']*1e4:>6.2f}bps  "
              f"net_ann={c['net_ann']*100:>6.2f}%")
else:
    print("(no _cache/ohlc.csv — run data.fetch_ohlc() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  inference must recover a PLANTED overnight-reversal knob and must NOT")
print("  manufacture significance when the true reversal is 0.")
print(f"  {'planted':>8} {'n':>6} {'b':>10} {'corr':>8} {'HAC_t':>8} {'p_plac':>8} {'win':>7}")
for rev in (0.0, 0.10):
    syn = data.synthetic_panel(reversal=rev, seed=376)
    s = st.summarize(syn)
    print(f"  {rev:>8.2f} {s['n']:>6,} {s['b']:>10.5f} {s['corr']:>8.3f} "
          f"{s['t']:>8.2f} {s['p_placebo']:>8.4f} {s['fade_win']*100:>6.1f}%")
