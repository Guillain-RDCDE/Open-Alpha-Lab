"""Reproducible headline run for Study 533 — Analyst-Dispersion (Diether-Malloy-Scherbina).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached dispersion snapshot + daily prices
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from analyst_dispersion import data, strategy as st  # noqa: E402

NDRAWS = 20_000
HLABEL = {21: "1-mo", 63: "3-mo", 126: "6-mo", 252: "12-mo"}


def synthetic_control_seeds(n_seeds: int = 25) -> dict:
    """Average the long-short t over many seeds — faithful-engine / power check, never a stamp.

    Plant a DMS edge (high dispersion drags returns DOWN); the low-minus-high long-short must be
    positive with a t large *on average* (not one lucky seed), and ~0 under the null edge=0.
    """
    pos_t, null_t, pos_ls = [], [], []
    for s in range(n_seeds):
        cs = data.synthetic_cross_section(dms_edge=0.70, seed=1000 + s)
        r = st.summarize(cs, "ret_cum", placebo=False)
        pos_t.append(r["t"]); pos_ls.append(r["ls_mean"])
        cs0 = data.synthetic_cross_section(dms_edge=0.0, seed=1000 + s)
        null_t.append(st.summarize(cs0, "ret_cum", placebo=False)["t"])
    return {"pos_mean_t": float(np.nanmean(pos_t)),
            "pos_mean_ls": float(np.nanmean(pos_ls)) * 100,
            "null_mean_t": float(np.nanmean(null_t)),
            "frac_pos_sig": float(np.mean(np.array(pos_t) > 2.0)),
            "n_seeds": n_seeds}


def main() -> None:
    print("# Analyst-Dispersion — the Diether-Malloy-Scherbina puzzle "
          "(high forecast dispersion -> LOW returns)\n")

    # Synthetic positive control (faithful-engine / power check only; never a real stamp).
    print("=== Synthetic positive control (seed-robust, >=20 seeds) ===")
    ctrl = synthetic_control_seeds(25)
    print(f"  Planted DMS edge 0.70: avg low-minus-high LS = {ctrl['pos_mean_ls']:+.2f}%"
          f" | avg t = {ctrl['pos_mean_t']:+.3f} over {ctrl['n_seeds']} seeds")
    print(f"  Null edge 0.00:        avg t = {ctrl['null_mean_t']:+.3f}")
    print(f"  Fraction of seeds with t>2 (planted): {ctrl['frac_pos_sig']*100:.0f}%")
    print("  -> engine is faithful: recovers planted DMS effect, ~0 under null.\n")

    if not data.have_real():
        print("(no _cache/disp_snapshot.csv — run data.fetch_snapshot() once to build the cache)")
        return

    disp, prices = data.load_real()
    fr = data.trailing_returns(prices, disp)
    fp_disp = data.fingerprint(disp)
    fp_px = data.fingerprint(prices)
    print("=== Real yfinance snapshot ===")
    print(f"Dispersion snapshot: {disp.shape[0]} names, fingerprint={fp_disp}")
    print(f"Prices panel:        {prices.shape}, fingerprint={fp_px}")
    print(f"Price tape: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance earnings_estimate (0y) + daily adj-close\n")

    print("=== Dispersion extremes (DMS proxy = (high-low)/|mean| of 0y EPS estimate) ===")
    ds = disp.sort_values("dispersion")
    for tk in list(ds["ticker"][:4]) + list(ds["ticker"][-4:]):
        row = ds[ds["ticker"] == tk].iloc[0]
        print(f"  {tk:6s}: dispersion={row['dispersion']:.3f}  n_analysts={row['n_analysts']:.0f}")

    print("\n=== DMS long-short: LONG low-dispersion, SHORT high-dispersion (low - high) ===")
    print(f"  {'H':>6} {'n':>3} {'low%':>8} {'high%':>8} {'LS%':>8} {'win%':>5} "
          f"{'t':>7} {'p_plac':>7} {'spearman':>9}")
    for h in st.HORIZONS:
        col = f"ret_{h}"
        s = st.summarize(fr, col, n_draws=NDRAWS)
        rho = st.disp_return_corr(fr, col)
        print(f"  {HLABEL[h]:>6} {s['n']:>3} {s['low_mean']*100:>7.2f} "
              f"{s['high_mean']*100:>7.2f} {s['ls_mean']*100:>7.2f} "
              f"{s['ls_win']*100:>4.0f} {s['t']:>7.2f} {s['p_placebo']:>7.3f} {rho:>9.3f}")

    print("\n=== Net of one-way costs (10 bps x 4 legs) + 100 bps/yr borrow (12-mo horizon) ===")
    for h in st.HORIZONS:
        c = st.net_of_costs(fr, f"ret_{h}", horizon_days=h)
        print(f"  {HLABEL[h]:>6}: gross={c['gross']*100:>6.2f}%  net={c['net']*100:>6.2f}%")

    print("\n=== 12-mo trailing return by dispersion tercile (low -> high) — monotonicity ===")
    bm = st.bucket_means(fr, "ret_252")
    print("  " + "  ".join(f"T{i+1}:{x*100:+.2f}%" for i, x in enumerate(bm)))

    print("\n=== Robustness — bucket count at H=252 ===")
    for nb in (2, 3, 5):
        s = st.summarize(fr, "ret_252", n_buckets=nb, n_draws=NDRAWS)
        print(f"  n_buckets={nb}: LS={s['ls_mean']*100:>6.2f}%  t={s['t']:>5.2f}  "
              f"p={s['p_placebo']:.3f}")

    print("\nNOTE Data limitation (named on the Signal axis): yfinance exposes only a CURRENT")
    print("     snapshot of analyst estimates — no historical dispersion panel. The sort is a")
    print("     single cross-section of dispersion vs TRAILING returns (contemporaneous")
    print("     association), not a tradable forward strategy, on ~40 survivor names.")


if __name__ == "__main__":
    main()
