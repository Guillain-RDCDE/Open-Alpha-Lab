"""Reproducible headline run for Study 442 — Anchored VWAP.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached 5-minute bars under ``_cache/`` if
present (the real-tape numbers, partial sessions dropped), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anchored_vwap import data, strategy as st

NDRAWS = 2000

print("# Anchored VWAP — does an anchored level act as a price magnet? (yfinance 5m bars)")
if data.have_real():
    bb = data.load_real_complete()
    s0 = st.session_avwap(bb["SPY"])
    span_first, span_last = s0["sess"].min(), s0["sess"].max()
    print(f"tickers        : {list(bb.keys())}")
    print(f"bars / ticker  : {len(bb['SPY'])}  (5-minute, RTH only, complete sessions)")
    print(f"sessions       : {s0['sess'].nunique()}  ({span_first} -> {span_last})")
    print(f"fingerprint    : {data.fingerprint(bb)}")

    print("\n# Pooled AVWAP magnet test — post-touch reversion (1-bar lag, 1 bp half-spread)")
    print(f"  {'H':>3} {'n':>6} {'av_bp':>7} {'ctrl_bp':>8} {'t_hac':>6} "
          f"{'t_vs_ctrl':>9} {'win%':>5} {'net_bp':>7}")
    for h in st.HORIZONS:
        r = st.run_experiment(bb, horizon=h)
        print(f"  {h:>3} {r['n_avwap']:>6} {r['avwap_mean']*1e4:>6.2f} "
              f"{r['ctrl_mean']*1e4:>7.2f} {r['t_hac']:>6.2f} {r['t_vs_ctrl']:>9.2f} "
              f"{r['win']*100:>4.0f} {r['net']*1e4:>7.2f}")

    print("\n# Random-level placebo at H=6 (is the AVWAP line any more special than a random "
          "level?)")
    r6 = st.run_experiment(bb, horizon=6, placebo=True, n_draws=NDRAWS)
    print(f"  AVWAP mean reaction = {r6['avwap_mean']*1e4:+.2f} bp ; "
          f"placebo p = {r6['p_placebo']:.3f}  "
          f"(share of random levels that match/beat AVWAP)")

    print("\n# Cost sweep at H=6 (half-spread, round trip on both legs)")
    av6 = []
    for tk, b in bb.items():
        av6.extend(st.avwap_reactions(st.session_avwap(b), 6).tolist())
    av6 = np.asarray(av6)
    for c in (0.5, 1.0, 2.0):
        nc = st.net_of_costs(av6, cost_bps=c)
        print(f"  half-spread={c:>3} bp: gross={nc['gross']*1e4:+.2f} bp  "
              f"net={nc['net']*1e4:+.2f} bp")

    print("\n# Per-ticker AVWAP mean reaction at H=6 (the dispersion — all noise)")
    for tk, b in bb.items():
        av = st.avwap_reactions(st.session_avwap(b), 6)
        print(f"  {tk:>4}: n={len(av):>4}  av={av.mean()*1e4:+.2f} bp  "
              f"t_hac={st.hac_t(av):+.2f}")
else:
    print("(no _cache/bars_SPY_5m.parquet — run data.fetch_basket() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the post-touch reversion test must light up when a magnet is PLANTED and must NOT")
print("  manufacture significance when the true magnet is 0.")
for m in (0.0, 0.10):
    px, _ = data.synthetic_panel(magnet=m, seed=442)
    av = st.avwap_reactions(st.session_avwap(px), 6)
    print(f"  magnet={m:+.2f}: n={len(av):>4}  av={av.mean()*1e4:+.2f} bp  "
          f"t_one={st.ttest_vs_zero(av):+.2f}  t_hac={st.hac_t(av):+.2f}  "
          f"win={(av > 0).mean()*100:.0f}%")
