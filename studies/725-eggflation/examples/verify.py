"""Reproducible headline run for Study 725 — "Eggflation" (trade the egg spike via CALM).

Prints every number quoted in docs/results.md and frozen into
notebooks/build_notebooks.py (the ``R`` dict). CALM + SPY come from the cached month-end
yfinance pull under ``_cache/``; the egg-price series is the hardcoded, cited,
**approximate** USDA/BLS retail series in :mod:`eggflation.data`. Deterministic & offline.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download CALM + SPY from Yahoo, then run
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

import numpy as np  # noqa: E402

from eggflation import data, strategy as st  # noqa: E402

try:
    from quantlab import repro  # noqa: E402
except Exception:  # pragma: no cover
    repro = None


def main(fetch: bool) -> None:
    if fetch:
        data.fetch_market(fetch=True)
    panel = data.build_panel(asof="2026-06-30")
    if panel.empty:
        print("No cached market data. Re-run with --fetch (needs network).")
        return

    print(f"\nEggflation — CALM vs a cited USDA/BLS egg-price proxy, "
          f"{panel.index.min().date()}..{panel.index.max().date()} ({len(panel)} months)\n")

    print("# The illusion — LEVEL correlation (two co-trending series)")
    lvl = float(np.corrcoef(panel["egg"], panel["calm"])[0, 1])
    print(f"  egg LEVEL vs CALM LEVEL corr : {lvl:+.3f}")

    print("\n# Does the egg-price CHANGE predict CALM's RETURN? (HAC t)")
    cm = st.contemporaneous(panel)
    print(f"  same-month  r_t   ~ g_t      : slope {cm['slope']:+.3f}  t {cm['t']:+.2f}  (UNTRADABLE)")
    for lag in (1, 2):
        pr = st.predictive(panel, lag=lag)
        print(f"  next-{lag}mo   r_t+{lag} ~ g_t      : slope {pr['slope']:+.3f}  t {pr['t']:+.2f}")
    p1s = st.predictive(panel, lag=1, control_spy=True)
    print(f"  next-1mo + market control     : slope {p1s['slope']:+.3f}  t {p1s['t']:+.2f}")

    print("\n# Placebo — circular-shift null for the predictive t")
    for lag in (1, 2):
        pl = st.circular_shift_placebo(panel, lag=lag, n_boot=5000, seed=725)
        print(f"  lag {lag}: obs t {pl['obs_t']:+.2f}  placebo p {pl['p']:.3f}  "
              f"null 95% [{pl['lo']:+.2f}, {pl['hi']:+.2f}]")

    print("\n# Reverse — does the STOCK lead the government egg print? (HAC t)")
    for lag in (1, 2):
        rv = st.reverse_lead(panel, lag=lag)
        print(f"  g_t+{lag} ~ r_t : slope {rv['slope']:+.3f}  t {rv['t']:+.2f}")

    print("\n# CALM's CAPM profile (is the 'egg stock' even market beta?)")
    ca = st.capm_alpha(panel["calm_ret"], panel["spy_ret"])
    print(f"  alpha {ca['alpha_m']*100:+.2f}%/mo (t {ca['t_alpha']:+.2f})  beta {ca['beta']:.2f}")

    print("\n# The egg-momentum timer (long CALM when egg momentum>0, 1-mo lag) vs benchmarks")
    print("# Sharpe = excess of a flat 2%/yr cash, both legs")
    tm = st.egg_momentum_timer(panel, window=3)
    net = st.apply_costs(tm, cost_bps_one_way=10.0)
    switches = int((tm["signal"].diff().abs() > 0).sum())
    for nm, r in [("egg timer (gross)", tm["timer"]), ("egg timer (net 10bp)", net),
                  ("buy & hold CALM", tm["calm"]), ("SPY", tm["spy"])]:
        s = st.summarize(r)
        print(f"  {nm:22s} Sharpe {s['sharpe']:.2f}  CAGR {s['cagr']*100:+.1f}%  "
              f"vol {s['vol']*100:.1f}%  maxDD {s['mdd']*100:.0f}%")
    print(f"  (long {tm['signal'].mean()*100:.0f}% of months, {switches} switches)")

    print("\n# Synthetic positive control — planted lead (beta=0.6) and null (beta=0)")
    w = data.synthetic_world(lead_beta=0.6)
    w0 = data.synthetic_world(lead_beta=0.0)
    pr = st.predictive(w, lag=1)
    pl = st.circular_shift_placebo(w, lag=1, n_boot=3000, seed=725)
    tmw = st.egg_momentum_timer(w)
    print(f"  planted lead: slope {pr['slope']:+.3f}  t {pr['t']:+.2f}  placebo p {pl['p']:.3f}  "
          f"timer Sharpe {st.summarize(tmw['timer'])['sharpe']:.2f} vs BH {st.summarize(tmw['calm'])['sharpe']:.2f}")
    print(f"  null world  : t {st.predictive(w0, lag=1)['t']:+.2f} (flat)")

    if repro is not None:
        print(f"\n{repro.data_stamp('egg+CALM+SPY monthly', panel[['egg', 'calm', 'spy']], asof='2026-06-30')}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
