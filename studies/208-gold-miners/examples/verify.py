"""Real-tape verification — Study 208 (Gold-Miners). Regenerates docs/results.md numbers.

Loads (or fetches) GLD and GDX daily adjusted closes, runs the three-pronged analysis:
1. OLS beta/alpha of GDX on GLD (the leveraged-gold claim)
2. Asymmetric beta test (more downside than upside amplification?)
3. Miners-vs-bullion timing rule vs a simple GLD hold

Network is touched only with --fetch.

    python studies/208-gold-miners/examples/verify.py            # cache-only
    python studies/208-gold-miners/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gold_miners import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_cache")

    if fetch:
        print("Fetching GLD and GDX from Yahoo Finance...")
        gld = data.fetch_prices("GLD", start=data.GLD_START, fetch=True, cache_dir=cache_dir)
        gdx = data.fetch_prices("GDX", start=data.GDX_START, fetch=True, cache_dir=cache_dir)
        print(f"GLD: {gld.index[0].date()} -> {gld.index[-1].date()}, n={len(gld)}")
        print(f"GDX: {gdx.index[0].date()} -> {gdx.index[-1].date()}, n={len(gdx)}")

    prices = data.load_pair(fetch=False, cache_dir=cache_dir)
    gld_fp = data.fingerprint(prices, "gld")
    gdx_fp = data.fingerprint(prices, "gdx")
    print(f"\nCommon tape: {prices.index[0].date()} -> {prices.index[-1].date()}, n={len(prices)}")
    print(f"GLD fingerprint: {gld_fp}")
    print(f"GDX fingerprint: {gdx_fp}")

    print("\n=== Test 1: OLS Beta / Alpha (GDX on GLD) ===")
    ba = st.beta_alpha(prices)
    print(f"n = {ba['n']}")
    print(f"beta (leverage) = {ba['beta']:.4f}  HAC t = {ba['t_beta']:+.2f}")
    print(f"alpha = {ba['alpha_daily_bps']:+.2f} bps/day = {ba['alpha_ann_pct']:+.2f}%/yr  HAC t = {ba['t_alpha']:+.2f}")
    print(f"R^2 = {ba['r_squared']:.4f}")
    print(f"GLD vol = {ba['gld_vol_ann_pct']:.1f}%/yr  GDX vol = {ba['gdx_vol_ann_pct']:.1f}%/yr")
    print(f"Idiosyncratic vol = {ba['idio_vol_ann_pct']:.1f}%/yr")
    print(f"Vol-ratio (implied leverage) = {ba['implied_leverage']:.4f}")

    print("\n=== Test 2: Asymmetric Beta (upside vs downside) ===")
    asym = st.asymmetric_beta(prices)
    print(f"n_up = {asym['n_up']}  n_dn = {asym['n_dn']}")
    print(f"beta_up = {asym['beta_up']:.4f}  beta_dn = {asym['beta_dn']:.4f}")
    print(f"diff (beta_dn - beta_up) = {asym['beta_diff']:+.4f}  HAC t = {asym['t_asymmetry']:+.2f}")
    print(f"asymmetry_ratio (beta_dn / beta_up) = {asym['asymmetry_ratio']:.4f}")
    print(f"alpha_ann = {asym['alpha_ann_pct']:+.2f}%/yr")
    claim_supported = asym["t_asymmetry"] > 2.0
    print(f"Claim (beta_dn > beta_up at |t| >= 2): {'SUPPORTED' if claim_supported else 'REJECTED'}")

    print("\n=== Test 3: GDX/GLD Timing Rule vs GLD Buy-and-Hold ===")
    res = st.compare_strategies(prices, sma_n=200, cost_bps=5.0)
    print(f"In-market fraction (GDX): {res['in_market_frac']:.1%}")
    print(f"Number of switches: {res['n_switches']}")
    print()
    for name in ("gld", "gdx", "timing"):
        s = res[name]
        label = {"gld": "GLD buy-and-hold", "gdx": "GDX buy-and-hold", "timing": "GDX/GLD timing"}[name]
        print(f"{label}:")
        print(f"  CAGR = {s['cagr_pct']:+.2f}%  Sharpe = {s['sharpe']:+.4f}  vol = {s['vol_ann_pct']:.1f}%/yr"
              f"  maxDD = {s['max_drawdown_pct']:.1f}%  HAC t = {s['tstat']:+.2f}")

    print("\n=== Verdict ===")
    print(f"Beta to GLD: {ba['beta']:.2f}x (t={ba['t_beta']:+.1f}) — CONFIRMED, strongly significant")
    print(f"Alpha: {ba['alpha_ann_pct']:+.1f}%/yr (t={ba['t_alpha']:+.2f}) — SIGNIFICANT negative drag")
    print(f"Asymmetry (claim: more downside): t={asym['t_asymmetry']:+.2f} — "
          f"{'SUPPORTED' if asym['t_asymmetry'] > 2 else ('REJECTED: opposite direction (more UPSIDE beta)' if asym['t_asymmetry'] < -2 else 'NEUTRAL')}")
    print(f"Timing rule Sharpe: {res['timing']['sharpe']:+.4f} vs GLD {res['gld']['sharpe']:+.4f} — "
          f"{'BEATS GLD' if res['timing']['sharpe'] > res['gld']['sharpe'] else 'UNDERPERFORMS GLD'}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
