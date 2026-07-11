"""Reproducible headline run for Study 655 — Ivy Portfolio.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached VTI/VEU/VNQ/AGG/DBC/BIL daily
closes under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))              # study package
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root (quantlab)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from ivy_portfolio import data, strategy as st  # noqa: E402

COST_BPS = 5.0
COST_BPS_HI = 10.0
N_SEEDS_RANDOM = 40
N_OUTER_SEEDS = 10
N_INNER_SEEDS = 20
SYN_MONTHS = 300
PLANTED_PERSISTENCE = 0.85


def fmt_perf(p):
    return (f"CAGR {p['cagr']*100:+.2f}%  vol {p['vol']*100:.2f}%  "
            f"Sharpe(excess of BIL) {p['sharpe']:.3f}  maxDD {p['max_dd']*100:.1f}%  (n={p['n']})")


print("# Ivy Portfolio — 20% each VTI/VEU/VNQ/AGG/DBC, with and without a 10-month SMA timer")

if not data.have_real():
    print("(cache miss — fetching VTI/VEU/VNQ/AGG/DBC/BIL once)")
    data.fetch()

px = data.load_prices()
px_asof = px[px.index <= data.AS_OF]
print(data_stamp("VTI/VEU/VNQ/AGG/DBC/BIL daily closes", px_asof, asof=data.AS_OF))

ret, close = data.monthly_panel()
print(f"monthly window: {ret.index.min().date()} -> {ret.index.max().date()} "
      f"({len(ret)} months, {len(ret)/12:.1f} years) — bound by BIL's 2007-05-30 inception "
      f"(the latest of the six)")

sig = st.sma_signal(close)
sig_live = sig.dropna(how="any")
print(f"10-month SMA signal live from {sig_live.index.min().date()} "
      f"({len(sig_live)} months) — 10-month warm-up + 1-month shift after the joint start")

rf = ret[data.CASH]

# --------------------------------------------------------------------------- #
# Headline arms
# --------------------------------------------------------------------------- #
sw = st.static_weights(ret.index)
static_net, static_turn = st.weighted_portfolio(ret, sw, cost_bps=COST_BPS)

b64w = st.sixty_forty_weights(ret.index)
b64_net, _ = st.weighted_portfolio(ret, b64w, cost_bps=COST_BPS)

vti = ret["VTI"]

tw = st.timed_weights(sig)
timed_net, timed_turn = st.weighted_portfolio(ret, tw, cost_bps=COST_BPS)
common = timed_net.index

print("\n# THE HEADLINE — 5-asset equal-weight Ivy vs 60/40 vs VTI (net, monthly rebal., "
      f"{COST_BPS:.0f} bps one-way)")
ps = st.perf(static_net, rf)
pb = st.perf(b64_net, rf)
pv = st.perf(vti, rf)
print(f"  Ivy static (20% x 5)     : {fmt_perf(ps)}")
print(f"  60/40 (VTI/AGG)          : {fmt_perf(pb)}")
print(f"  VTI buy-and-hold         : {fmt_perf(pv)}")

a_64 = st.active_stats(static_net, b64_net)
a_vti = st.active_stats(static_net, vti)
print(f"  active Ivy-60/40 : {a_64['mean_bps']:+.1f} bps/mo = {a_64['ann_pct']:+.2f}%/yr  "
      f"HAC t = {a_64['hac_t']:+.2f}")
print(f"  active Ivy-VTI   : {a_vti['mean_bps']:+.1f} bps/mo = {a_vti['ann_pct']:+.2f}%/yr  "
      f"HAC t = {a_vti['hac_t']:+.2f}")

boot_64 = st.bootstrap_sharpe_diff(static_net, b64_net, rf=rf, seed=655)
boot_vti = st.bootstrap_sharpe_diff(static_net, vti, rf=rf, seed=655)
print(f"  bootstrap Sharpe diff (Ivy - 60/40): {boot_64['point']:+.3f}  "
      f"95% CI [{boot_64['ci95'][0]:+.3f}, {boot_64['ci95'][1]:+.3f}]  "
      f"(Ivy wins {boot_64['frac_a_wins']*100:.1f}% of {boot_64['n_boot']} resamples)")
print(f"  bootstrap Sharpe diff (Ivy - VTI)  : {boot_vti['point']:+.3f}  "
      f"95% CI [{boot_vti['ci95'][0]:+.3f}, {boot_vti['ci95'][1]:+.3f}]  "
      f"(Ivy wins {boot_vti['frac_a_wins']*100:.1f}% of {boot_vti['n_boot']} resamples)")

# --------------------------------------------------------------------------- #
# The 10-month SMA overlay — risk reduction or alpha?
# --------------------------------------------------------------------------- #
print(f"\n# THE 10-MONTH SMA OVERLAY — timed vs static Ivy, same {len(common)}-month window")
ps_c = st.perf(static_net.loc[common], rf)
pb_c = st.perf(b64_net.loc[common], rf)
pt = st.perf(timed_net, rf)
print(f"  Ivy static (this window) : {fmt_perf(ps_c)}")
print(f"  60/40 (this window)      : {fmt_perf(pb_c)}")
print(f"  Ivy TIMED (10-mo SMA)    : {fmt_perf(pt)}")
print(f"  drawdown ratio timed/static = {pt['max_dd']/ps_c['max_dd']:.2f}   "
      f"drawdown ratio timed/60-40 = {pt['max_dd']/pb_c['max_dd']:.2f}")

a_ts = st.active_stats(timed_net, static_net.loc[common])
a_t64 = st.active_stats(timed_net, b64_net.loc[common])
print(f"  active timed-static : {a_ts['mean_bps']:+.1f} bps/mo = {a_ts['ann_pct']:+.2f}%/yr  "
      f"HAC t = {a_ts['hac_t']:+.2f}")
print(f"  active timed-60/40  : {a_t64['mean_bps']:+.1f} bps/mo = {a_t64['ann_pct']:+.2f}%/yr  "
      f"HAC t = {a_t64['hac_t']:+.2f}")

boot_ts = st.bootstrap_sharpe_diff(timed_net, static_net.loc[common], rf=rf, seed=655)
boot_t64 = st.bootstrap_sharpe_diff(timed_net, b64_net.loc[common], rf=rf, seed=655)
print(f"  bootstrap Sharpe diff (timed - static): {boot_ts['point']:+.3f}  "
      f"95% CI [{boot_ts['ci95'][0]:+.3f}, {boot_ts['ci95'][1]:+.3f}]  "
      f"(timed wins {boot_ts['frac_a_wins']*100:.1f}%)")
print(f"  bootstrap Sharpe diff (timed - 60/40) : {boot_t64['point']:+.3f}  "
      f"95% CI [{boot_t64['ci95'][0]:+.3f}, {boot_t64['ci95'][1]:+.3f}]  "
      f"(timed wins {boot_t64['frac_a_wins']*100:.1f}%)")

print("\n# Per-asset standalone — which legs actually dragged the composite?")
for tkr in data.ASSETS:
    p = st.perf(ret[tkr], rf)
    print(f"  {tkr:<4}: {fmt_perf(p)}")

print("\n# Sub-period — does the diversification/timing story live off 2008 alone? (ex-GFC "
      "2007-07 -> 2009-06)")


def ex_gfc(s: pd.Series) -> pd.Series:
    return s[(s.index < "2007-07-01") | (s.index > "2009-06-30")]


sg, bg, rfg = ex_gfc(static_net), ex_gfc(b64_net), ex_gfc(rf)
ps_g, pb_g = st.perf(sg, rfg), st.perf(bg, rfg)
a_g = st.active_stats(sg, bg)
print(f"  Ivy static ex-GFC : {fmt_perf(ps_g)}")
print(f"  60/40 ex-GFC      : {fmt_perf(pb_g)}")
print(f"  active Ivy-60/40 ex-GFC: {a_g['ann_pct']:+.2f}%/yr  HAC t = {a_g['hac_t']:+.2f}")

tg = ex_gfc(timed_net)
common_g = tg.index
sg2, pt_g = static_net.loc[common_g], st.perf(tg, rf)
ps_g2 = st.perf(sg2, rf)
a_ts_g = st.active_stats(tg, sg2)
a_t64_g = st.active_stats(tg, ex_gfc(b64_net.loc[common]))
print(f"  Ivy timed  ex-GFC : {fmt_perf(pt_g)}")
print(f"  Ivy static ex-GFC (timed window): {fmt_perf(ps_g2)}")
print(f"  active timed-static ex-GFC: {a_ts_g['ann_pct']:+.2f}%/yr  HAC t = {a_ts_g['hac_t']:+.2f}")
print(f"  active timed-60/40  ex-GFC: {a_t64_g['ann_pct']:+.2f}%/yr  HAC t = {a_t64_g['hac_t']:+.2f}")

print("\n# Costs & turnover — one-way bps x NAV on total absolute weight change")
static_ann_turn = float(static_turn.mean() * 12)
timed_ann_turn = float(timed_turn.mean() * 12)
flips = (sig_live != sig_live.shift(1)).iloc[1:]
flips_per_yr = float(flips.values.sum() / (len(sig_live) / 12))
print(f"  Ivy static turnover : {static_ann_turn*100:.0f}%/yr one-way (drift-only rebalance)")
print(f"  Ivy timed  turnover : {timed_ann_turn*100:.0f}%/yr one-way "
      f"({flips_per_yr:.2f} sleeve-flips/yr across the 5 legs)")
for cb in (COST_BPS, COST_BPS_HI):
    tn, _ = st.weighted_portfolio(ret, tw, cost_bps=cb)
    p = st.perf(tn, rf)
    print(f"  timed cost={cb:>4.1f} bps: CAGR {p['cagr']*100:+.2f}%  "
          f"Sharpe {p['sharpe']:.3f}  maxDD {p['max_dd']*100:.1f}%")

# --------------------------------------------------------------------------- #
# Third axis — matched-exposure random-timing control (>= 20 seeds)
# --------------------------------------------------------------------------- #
print(f"\n# THIRD AXIS — does the 10-month SMA carry real timing information, "
      f"or just less exposure? ({N_SEEDS_RANDOM} matched-exposure random-timing seeds)")
rb = st.random_timing_baseline(ret, sig, cost_bps=COST_BPS, n_seeds=N_SEEDS_RANDOM, base_seed=655)
print(f"  actual timed  : {fmt_perf(rb['actual'])}")
print(f"  random-timing baseline (same time-in-market per sleeve, shuffled WHEN): "
      f"mean CAGR {rb['base_cagr_mean']*100:+.2f}%  mean Sharpe {rb['base_sharpe_mean']:.3f}  "
      f"mean maxDD {rb['base_dd_mean']*100:.1f}%")
print(f"  actual beats {rb['dd_beat_share']*100:.0f}% of {rb['n_seeds']} shuffles on drawdown, "
      f"{rb['sharpe_beat_share']*100:.0f}% on Sharpe")

# --------------------------------------------------------------------------- #
# Synthetic machinery control — deterministic, no network (never cited for the stamp)
# --------------------------------------------------------------------------- #
print("\n# Synthetic machinery control — deterministic, no network")
print("  the exposure-matched detector must fire at ~chance under a null (persistence=0)")
print("  and clearly above chance once a trend is planted (persistence>0)")


def _synthetic_beat_shares(persistence: float, n_outer: int, n_inner: int) -> tuple[float, float]:
    sh, dd = [], []
    for s in range(n_outer):
        panel = data.synthetic_world(persistence=persistence, seed=655 + s, n_months=SYN_MONTHS)
        closes = panel[data.ASSETS]
        rets = pd.DataFrame(index=panel.index)
        rets[data.ASSETS] = closes.pct_change()
        rets[data.CASH] = panel[data.CASH + "_lvl"].pct_change()
        rets = rets.dropna(how="all")
        sig_syn = st.sma_signal(closes)
        r = st.random_timing_baseline(rets, sig_syn, cost_bps=COST_BPS, n_seeds=n_inner,
                                      base_seed=1000 + s)
        sh.append(r["sharpe_beat_share"])
        dd.append(r["dd_beat_share"])
    return float(np.mean(sh)), float(np.mean(dd))


null_sh, null_dd = _synthetic_beat_shares(0.0, N_OUTER_SEEDS, N_INNER_SEEDS)
print(f"  null (persistence=0.00), {N_OUTER_SEEDS} worlds x {N_INNER_SEEDS} shuffles: "
      f"mean Sharpe-beat-share {null_sh*100:.1f}%  mean DD-beat-share {null_dd*100:.1f}%  "
      "(chance = 50%)")

planted_sh, planted_dd = _synthetic_beat_shares(PLANTED_PERSISTENCE, N_OUTER_SEEDS, N_INNER_SEEDS)
print(f"  planted (persistence={PLANTED_PERSISTENCE}), {N_OUTER_SEEDS} worlds x {N_INNER_SEEDS} "
      f"shuffles: mean Sharpe-beat-share {planted_sh*100:.1f}%  "
      f"mean DD-beat-share {planted_dd*100:.1f}%")
