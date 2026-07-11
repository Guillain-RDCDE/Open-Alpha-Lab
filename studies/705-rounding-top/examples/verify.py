"""Reproducible headline run for Study 705 — Rounding Top (dome distribution).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first (reads the parquet OHLC panel
under ``_cache/`` built once by ``data.fetch_panel``). The synthetic positive control
runs with no network.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # refresh the cache from yfinance first
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rounding_top import data, strategy as st  # noqa: E402

NDRAWS = 5_000


def main() -> None:
    if "--fetch" in sys.argv:
        print("# fetching real OHLC panel from yfinance ...")
        data.fetch_panel()

    print("# Rounding Top — mechanical dome-distribution breakdown detector on "
          "SPY + 29 large-caps (yfinance)")
    if data.have_real():
        panel = data.load_real()
        spans = [b.index for b in panel.values()]
        lo = min(s.min() for s in spans).date()
        hi = max(s.max() for s in spans).date()
        years = (max(s.max() for s in spans) - min(s.min() for s in spans)).days / 365.25
        print(f"panel        : {len(panel)} names, {lo} -> {hi} ({years:.1f} years)")
        print(f"fingerprint  : {data.fingerprint(panel)}")

        print("\n# Confirmed-breakdown forward SHORT returns vs the every-bar SHORT base rate")
        print("  (short at t+1 open, cover close t+1+H; 5 bps one-way x2 + 30 bps/yr borrow, "
              "pro-rated)")
        print(f"  {'H':>3} {'n_sig':>6} {'names':>5} {'gross%':>7} {'net%':>7} "
              f"{'base%':>7} {'win%':>5} {'bwin%':>5} {'t0':>6} {'tHAC':>6} "
              f"{'tBase':>6} {'p_plac':>7}")
        for h in st.HORIZONS:
            r = st.run_experiment(panel, base_len=90, horizon=h, cost_bps=5.0,
                                  borrow_bps=30.0, n_draws=NDRAWS, seed=705)
            print(f"  {h:>3} {r['n_signals']:>6} {r['n_names']:>5} "
                  f"{r['gross_mean']*100:>6.2f} {r['net_mean']*100:>6.2f} "
                  f"{r['base_mean']*100:>6.2f} {r['win']*100:>4.0f} "
                  f"{r['base_win']*100:>4.0f} {r['t_zero']:>6.2f} {r['t_hac']:>6.2f} "
                  f"{r['t_vs_base']:>6.2f} {r['p_placebo']:>7.3f}")

        print("\n# Robustness at H=20 — vary the dome window length (base_len)")
        for bl in (60, 90, 120):
            r = st.run_experiment(panel, base_len=bl, horizon=20, cost_bps=5.0,
                                  borrow_bps=30.0, n_draws=NDRAWS, seed=705)
            print(f"  base_len={bl:>3}: n={r['n_signals']:>4}  gross={r['gross_mean']*100:>5.2f}%"
                  f"  t0={r['t_zero']:>5.2f}  tBase={r['t_vs_base']:>5.2f}  p={r['p_placebo']:.3f}")

        print("\n# Robustness at H=20, base_len=90 — vary the dome fit threshold (r2_min)")
        for r2 in (0.45, 0.55, 0.70):
            r = st.run_experiment(panel, base_len=90, horizon=20, r2_min=r2, cost_bps=5.0,
                                  borrow_bps=30.0, n_draws=NDRAWS, seed=705)
            print(f"  r2_min={r2:>4}: n={r['n_signals']:>4}  gross={r['gross_mean']*100:>5.2f}%"
                  f"  t0={r['t_zero']:>5.2f}  tBase={r['t_vs_base']:>5.2f}  p={r['p_placebo']:.3f}")
    else:
        print("(no _cache/ panel — run `python examples/verify.py --fetch` once to build it)")

    print("\n# Synthetic positive control — deterministic, no network")
    print("  edge=0 plants the SHAPE but no continuation (a true null: the figure fires but")
    print("  must NOT beat the base rate); edge<0 plants a real post-breakdown decline.")
    for edge in (0.0, -0.10):
        panel, _ = data.synthetic_panel(edge=edge, seed=705)
        r = st.run_experiment(panel, base_len=90, horizon=20, cost_bps=0.0, borrow_bps=0.0,
                              n_draws=2_000, seed=705)
        print(f"  edge={edge:+.2f}: n={r['n_signals']:>4}  gross={r['gross_mean']*100:>6.2f}%"
              f"  base={r['base_mean']*100:>5.2f}%  t0={r['t_zero']:>6.2f}"
              f"  tBase={r['t_vs_base']:>6.2f}  p={r['p_placebo']:.3f}  win={r['win']*100:.0f}%")

    print("\n# Null robustness — 20 seeds, edge=0, H=20 (the null must NOT fire)")
    ts = []
    for i in range(20):
        seed = 705 + i
        pnl, _ = data.synthetic_panel(edge=0.0, seed=seed)
        r = st.run_experiment(pnl, base_len=90, horizon=20, cost_bps=0.0, borrow_bps=0.0,
                              placebo=False)
        ts.append(r["t_vs_base"])
    import numpy as np  # noqa: E402
    ts = np.asarray(ts)
    print(f"  mean tBase = {ts.mean():+.2f}  (sd {ts.std(ddof=1):.2f}),  "
          f"|t| >= 2 in {(np.abs(ts) >= 2).sum()}/20 seeds")


if __name__ == "__main__":
    main()
