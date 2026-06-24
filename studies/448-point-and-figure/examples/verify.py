"""Reproducible headline run for Study 448 — Point & Figure price targets.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under
``_cache/`` if present (the real-tape numbers) and always runs the synthetic control
with no network. The in-progress final bar is dropped (as-of = 2026-06-23).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from point_and_figure import data, strategy as st  # noqa: E402

N_DRAWS = 2000
TICKERS = list(data.DEFAULT_TICKERS)


def pooled(side: str, n_draws: int = N_DRAWS):
    hit_ind, net = [], []
    th = tn = bh = bt = 0
    per = []
    for t in TICKERS:
        if not data.have_real(t):
            continue
        b = data.load_real(t, fetch=False).iloc[:-1]      # drop in-progress bar
        r = st.run_experiment(b, n_draws=n_draws, side=side)
        if r["n_signals"] == 0:
            continue
        L = r["ledger"]; p = r["pnl"]; base = r["baseline"]["baseline_hit_rate"]
        hit_ind += list((L["outcome"] == "hit").astype(float))
        net.append(p["net_sample"])
        th += p["hit_rate"] * r["n_signals"]; tn += r["n_signals"]
        bh += base * r["baseline"]["n_baseline"]; bt += r["baseline"]["n_baseline"]
        per.append((t, r["n_signals"], p["hit_rate"] * 100, base * 100,
                    p["net_mean"] * 100, p["t_net"], r["placebo"]["p_value"]))
    net = np.concatenate(net) if net else np.array([])
    hit_ind = np.array(hit_ind)
    return dict(hit=th / tn * 100 if tn else float("nan"),
                base=bh / bt * 100 if bt else float("nan"), n=tn,
                net=net, hit_ind=hit_ind, per=per)


print("# Point & Figure — horizontal-count targets on 4 daily tapes (yfinance)")
print("# box = 2% of median price, 3-box reversal, double-top buy / double-bottom sell")

if all(data.have_real(t) for t in TICKERS):
    b = data.load_real("SPY", fetch=False)
    print(f"as-of (in-progress bar dropped): {b.iloc[:-1].index.max().date()}  "
          f"SPY fingerprint: {data.fingerprint(b.iloc[:-1])}")

    pb = pooled("buy")
    ps = pooled("sell")

    print("\n# Per-instrument (BUY): ticker, n, hit%, baseline%, net%, t_net(HAC), placebo_p")
    for r in pb["per"]:
        print(f"  {r[0]:>5} n={r[1]:>3}  hit={r[2]:>5.1f}%  base={r[3]:>5.1f}%  "
              f"net={r[4]:>6.2f}%  t={r[5]:>5.2f}  p={r[6]:.3f}")

    print("\n# Pooled hit-rate edge over the same-distance baseline (the SIGNAL test)")
    for name, P in (("BUY ", pb), ("SELL", ps)):
        edge = P["hit"] - P["base"]
        t_edge = st.ttest_vs_zero(P["hit_ind"] - P["base"] / 100.0)
        t_net = st.hac_t(P["net"])
        print(f"  {name}: n={P['n']:>3}  hit={P['hit']:>5.1f}%  base={P['base']:>5.1f}%  "
              f"edge=+{edge:>4.1f}pp  t(hit-base)={t_edge:>5.2f}  "
              f"net={P['net'].mean()*100:>6.2f}%  t_net={t_net:>5.2f}  "
              f"win={ (P['net']>0).mean()*100:>4.1f}%")

    print("\n# Robustness — box-size sweep (pooled BUY hit% vs baseline%)")
    for bf in (0.015, 0.02, 0.03):
        th = tn = bh = bt = 0
        for t in TICKERS:
            bb = data.load_real(t, fetch=False).iloc[:-1]
            r = st.run_experiment(bb, box_frac=bf, n_draws=800, side="buy")
            if r["n_signals"] == 0:
                continue
            th += r["pnl"]["hit_rate"] * r["n_signals"]; tn += r["n_signals"]
            bh += r["baseline"]["baseline_hit_rate"] * r["baseline"]["n_baseline"]
            bt += r["baseline"]["n_baseline"]
        print(f"  box_frac={bf}: n={tn:>3}  hit={th/tn*100:>5.1f}%  base={bh/bt*100:>5.1f}%")
else:
    print("(no _cache/ tapes — run data.load_real(t, fetch=True) once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must recover a PLANTED continuation and must NOT manufacture a")
print("  hit-rate edge over the same-distance baseline when the true edge is 0.")
for edge in (0.0, 0.6):
    b, _ = data.synthetic_panel(edge=edge, seed=448)
    r = st.run_experiment(b, n_draws=1500, side="buy")
    if r["n_signals"] == 0:
        print(f"  planted edge={edge:+.2f}: no signals")
        continue
    print(f"  planted edge={edge:+.2f}: nsig={r['n_signals']:>4}  "
          f"hit={r['pnl']['hit_rate']*100:>5.1f}%  "
          f"base={r['baseline']['baseline_hit_rate']*100:>5.1f}%  "
          f"net={r['pnl']['net_mean']*100:>6.2f}%  t={r['pnl']['t_net']:>5.2f}  "
          f"placebo_p={r['placebo']['p_value']:.3f}")
