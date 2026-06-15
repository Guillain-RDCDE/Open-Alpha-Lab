"""Reproduce the real run (docs/results.md) — Study 201 (Dividend-Growth).

Loads the cached yfinance panel (40 large-cap survivors, annual total return + dividend
streak), runs the three-arm analysis (growers vs EW basket vs high-yield non-growers),
and prints the headline numbers that feed docs/results.md.

    python examples/verify.py           # cache-only (no network)
    python examples/verify.py --fetch   # re-fetch from yfinance and repopulate cache
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

from dividend_growth import data, strategy as st  # noqa: E402


def main(fetch: bool = False) -> None:
    print("Study 201 — Dividend-Growth — real panel\n")

    panel = data.load_real_panel(fetch=fetch)
    if panel["streaks"].empty:
        print("No cached data found.  Run with --fetch to populate the cache.")
        return

    streaks = panel["streaks"]
    fwd_ret = panel["fwd_ret"]
    ew_ret = panel["ew_ret"]
    n_tickers = len(panel["tickers"])

    # --- growers arm ---
    grower_mask = st.classify_growers(streaks, streak_years=3)
    grower_ret = st.arm_returns(fwd_ret, grower_mask)

    # --- high-yield non-grower arm ---
    hynon_mask = st.classify_high_yield_nongrowers(streaks, div_yield=None, streak_years=3)
    hynon_ret = st.arm_returns(fwd_ret, hynon_mask)

    # --- EW baseline ---
    ew_aligned = ew_ret.reindex(grower_ret.index)

    # --- hedge ---
    spread = st.hedge_returns(grower_ret, ew_aligned)

    # --- summaries ---
    sg = st.summary(grower_ret)
    se = st.summary(ew_aligned)
    sh = st.summary(spread)
    sn = st.summary(hynon_ret)
    sn_vs_ew = st.summary(st.hedge_returns(hynon_ret, ew_ret.reindex(hynon_ret.index)))

    fp = data.fingerprint(panel)

    print(f"Universe: {n_tickers} tickers (survivorship-biased — named on Signal axis)")
    print(f"Years: {streaks.index.min()} – {streaks.index.max()}  |  Panel fingerprint: {fp}\n")

    print(f"{'Arm':<22} {'n_yrs':>6} {'mean':>8} {'vol':>8} {'Sharpe':>8} {'t':>7} {'hit%':>7}")
    print("-" * 70)
    for lbl, s in [("Growers (3yr streak)", sg), ("EW basket", se),
                   ("High-yield non-growers", sn)]:
        n = s["n"]
        m = f"{s['mean']:+.2%}" if s["mean"] == s["mean"] else "  n/a"
        v = f"{s['vol']:.2%}" if s["vol"] == s["vol"] else "  n/a"
        sr = f"{s['sharpe']:+.2f}" if s["sharpe"] == s["sharpe"] else "  n/a"
        t = f"{s['tstat']:+.2f}" if s["tstat"] == s["tstat"] else "  n/a"
        h = f"{s['hit_rate']:.0%}" if s["hit_rate"] == s["hit_rate"] else "  n/a"
        print(f"{lbl:<22} {n:>6} {m:>8} {v:>8} {sr:>8} {t:>7} {h:>7}")

    print()
    print(f"Grower spread vs EW:  mean {sh['mean']:+.2%}/yr  t = {sh['tstat']:+.2f}  (n={sh['n']} years)")
    print(f"HY-nongrower vs EW:   mean {sn_vs_ew['mean']:+.2%}/yr  t = {sn_vs_ew['tstat']:+.2f}  (n={sn_vs_ew['n']} years)")
    print()
    grower_counts = grower_mask.sum(axis=1)
    print(f"Growers per year: min={grower_counts.min()}, median={grower_counts.median():.0f}, max={grower_counts.max()}")

    print("\nVerdict:")
    t = sh["tstat"]
    if t == t and abs(t) >= 2.0 and sh["mean"] > 0:
        print("  Signal: REAL — growers significantly outperform the EW basket")
    elif t == t and abs(t) >= 1.5:
        print("  Signal: WEAK — some tendency but below the inference bar")
    else:
        print("  Signal: NONE/WEAK — grower spread is indistinguishable from zero noise")
    print("  Tradability: MIRAGE — survivorship bias + annual turnover costs + capacity constraints")


if __name__ == "__main__":
    main("--fetch" in sys.argv)
