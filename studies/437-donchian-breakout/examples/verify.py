"""Real-tape verification — Study 437 (Donchian-Breakout).

Reproduces every number in docs/results.md. Cache-first (offline once cached);
network is touched only with --fetch.

    python studies/437-donchian-breakout/examples/verify.py            # cache-only
    python studies/437-donchian-breakout/examples/verify.py --fetch    # refresh tapes

The headline instrument is SPY; the panel probes generality. The 20-day Donchian
breakout is raced (NET, excess-vs-excess of the same flat-cash baseline) against
buy-and-hold, an SMA(20) cross benchmark, and a matched random-timing control, with
HAC t-stats and a block-permutation placebo.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from donchian_breakout import data, strategy as st  # noqa: E402

N = 20
COST = 2.0  # one-way bps


def main(fetch: bool) -> None:
    if fetch:
        for t in data.PANEL:
            b = data.load_real(t, fetch=True)
            print(f"{t:4s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    spy = data.load_real("SPY")
    print("=== data stamp (SPY) ===")
    print(f"SPY {spy.index[0].date()}..{spy.index[-1].date()} "
          f"n={len(spy)} fp={data.fingerprint(spy)}\n")

    # --- headline: the four arms on SPY, net of 2 bps one-way ---
    exp = st.run_experiment(spy, n=N, cost_bps=COST, allow_short=False, random_seed=42)
    print("=== SPY: Donchian-20 long/flat vs BH vs SMA(20) vs random (net 2bps) ===")
    for k in ("bh", "donchian", "sma", "random"):
        s = exp[k]
        print(f"{k:9s} CAGR={s['cagr']:+.2%} Sharpe={s['sharpe']:+.3f} "
              f"vol={s['vol_ann']:.2%} maxDD={s['max_drawdown']:+.2%} t={s['tstat']:+.2f}")
    print(f"in-market frac = {exp['in_market_frac']:.1%}, switches = {exp['n_switches']}")
    print(f"Sharpe-diff HAC t: Donchian vs BH     = {exp['t_vs_bh']:+.2f}")
    print(f"Sharpe-diff HAC t: Donchian vs random = {exp['t_vs_random']:+.2f}")
    print(f"Sharpe-diff HAC t: Donchian vs SMA    = {exp['t_vs_sma']:+.2f}")

    # --- permutation placebo (gross, to test the timing itself) ---
    sig = st.breakout_signal(spy, n=N, allow_short=False)
    perm = st.permutation_pvalue(spy, sig, n_perm=2000, block=N, cost_bps=0.0, seed=437)
    print(f"\n=== block-permutation placebo (gross, block={perm['block']}) ===")
    print(f"real mean = {perm['real_mean_bps']:+.3f} bps/day, "
          f"p-value = {perm['p_value']:.4f} ({perm['n_perm']} perms)")

    # --- cost sweep (net Sharpe) ---
    print("\n=== cost sweep (SPY Donchian-20 long/flat, net Sharpe) ===")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        e = st.run_experiment(spy, n=N, cost_bps=c, random_seed=42)
        print(f"cost={c:5.1f}bps  Sharpe={e['donchian']['sharpe']:+.3f}  "
              f"CAGR={e['donchian']['cagr']:+.2%}  t_vs_bh={e['t_vs_bh']:+.2f}")

    # --- window robustness ---
    print("\n=== window robustness (SPY long/flat, net 2bps) ===")
    for w in (10, 20, 55, 100):
        e = st.run_experiment(spy, n=w, cost_bps=COST, random_seed=42)
        print(f"N={w:3d}  Sharpe={e['donchian']['sharpe']:+.3f}  "
              f"BH={e['bh']['sharpe']:+.3f}  t_vs_bh={e['t_vs_bh']:+.2f}")

    # --- long/short variant ---
    e_ls = st.run_experiment(spy, n=N, cost_bps=COST, allow_short=True, random_seed=42)
    print("\n=== SPY Donchian-20 long/SHORT (net 2bps, borrow 50bps/yr) ===")
    print(f"Sharpe={e_ls['donchian']['sharpe']:+.3f}  CAGR={e_ls['donchian']['cagr']:+.2%}  "
          f"maxDD={e_ls['donchian']['max_drawdown']:+.2%}  t_vs_bh={e_ls['t_vs_bh']:+.2f}")

    # --- panel breakdown ---
    print("\n=== panel (Donchian-20 long/flat, net 2bps): Sharpe vs BH ===")
    for t in data.PANEL:
        b = data.load_real(t)
        e = st.run_experiment(b, n=N, cost_bps=COST, random_seed=42)
        print(f"{t:4s} n={e['donchian']['n_days']:5d} "
              f"Donchian={e['donchian']['sharpe']:+.3f} BH={e['bh']['sharpe']:+.3f} "
              f"t_vs_bh={e['t_vs_bh']:+.2f}")

    # --- synthetic positive control ---
    print("\n=== synthetic control (planted-edge knob) ===")
    for edge, lbl in ((0.0, "null (edge=0)"), (0.4, "trend (edge=0.4)")):
        sbars, truth = data.synthetic_panel(n_days=6000, edge=edge, seed=437)
        e = st.run_experiment(sbars, n=N, cost_bps=COST, random_seed=42)
        sig_s = st.breakout_signal(sbars, n=N, allow_short=False)
        perm_s = st.permutation_pvalue(sbars, sig_s, n_perm=2000, block=N, seed=437)
        print(f"{lbl:18s} Donchian Sharpe={e['donchian']['sharpe']:+.3f} "
              f"BH={e['bh']['sharpe']:+.3f} t_vs_bh={e['t_vs_bh']:+.2f} "
              f"perm_p={perm_s['p_value']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
