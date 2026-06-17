"""Cache-only verification -- Study 288 (Ramadan-Effect). Regenerates docs/results.md numbers.

Loads the cached MENA-proxy (iShares MSCI Saudi Arabia, KSA) + S&P 500 monthly panel
(``_cache/ramadan_proxy.parquet``), labels each month against the hardcoded Ramadan
windows, and runs the full analysis: Ramadan-vs-rest means, HAC (Newey-West) t, Welch t,
permutation test, the S&P placebo, the power calculation, and the tradable backtest.

Cache-only: no network. Populate the cache once with
``python -c "import sys;sys.path.insert(0,'studies/288-ramadan-effect');from ramadan_effect import data;data.fetch_proxy(fetch=True)"``.

    python studies/288-ramadan-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from ramadan_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 288 -- Ramadan-Effect -- cache-only verification\n")
    print("Data: iShares MSCI Saudi Arabia (KSA) + S&P 500 monthly (cache-only)")
    print("      Ramadan windows: hardcoded in data.py (Umm al-Qura, 2000-2026)\n")

    try:
        px = data.fetch_proxy(fetch=False)
    except FileNotFoundError:
        print("No cache present. Run data.fetch_proxy(fetch=True) once to populate it.")
        return

    df = data.proxy_returns(px)
    fp = data.fingerprint(df)
    print(f"Data stamp: {df.index.min().date()} .. {df.index.max().date()}, "
          f"n={len(df)} months, {int(df['ramadan'].sum())} Ramadan months, fingerprint={fp}\n")

    g = st.ramadan_comparison(df, "mena_ret", n_permutations=10_000, seed=288)
    p = st.ramadan_comparison(df, "spx_ret", n_permutations=10_000, seed=288)
    bt = st.ramadan_timing_backtest(df, "mena_ret", one_way_bps=30.0)

    print("=== MENA proxy (KSA): Ramadan vs rest ===")
    print(f"  n Ramadan months:  {g['n_ram']}   n rest: {g['n_rest']}")
    print(f"  Annualised return: Ramadan {g['ann_ram']*100:+.1f}%/yr  vs  rest {g['ann_rest']*100:+.1f}%/yr")
    print(f"  Mean gap:          {g['diff']*1e4:+.1f} bps/mo")
    print(f"  Annualised vol:    Ramadan {g['vol_ram']*100:.1f}%   rest {g['vol_rest']*100:.1f}%")
    print(f"  HAC dummy t:       {g['nw_t_diff']:+.3f}")
    print(f"  Welch t (p):       {g['welch_t']:+.3f} ({g['welch_p']:.4f})")
    print(f"  Permutation p:     {g['perm_p']:.4f}")
    print(f"  Up-month rate:     Ramadan {g['hit_ram']*100:.1f}%   rest {g['hit_rest']*100:.1f}%\n")

    print("=== S&P 500 placebo: Ramadan vs rest ===")
    print(f"  Mean gap:      {p['diff']*1e4:+.1f} bps/mo")
    print(f"  HAC dummy t:   {p['nw_t_diff']:+.3f}")
    print(f"  Permutation p: {p['perm_p']:.4f}")
    print("  (A clean placebo: no positive Ramadan effect on the S&P.)\n")

    vol_mo = float(df["mena_ret"].std(ddof=1))
    mdd = st.min_detectable_diff(vol_mo, g["n_ram"], g["n_rest"])
    print("=== The n=11 power problem ===")
    print(f"  Monthly vol: {vol_mo*100:.2f}%   n_ram={g['n_ram']}  n_rest={g['n_rest']}")
    print(f"  Min detectable gap (80% power): {mdd*1e4:.0f} bps/mo")
    print(f"  Observed gap: {abs(g['diff'])*1e4:.0f} bps/mo "
          f"({abs(g['diff'])/mdd:.0%} of the detection floor)\n")

    print("=== Tradable backtest: long MENA in Ramadan, flat otherwise ===")
    print(f"  Buy & hold:           {bt['bah_ann']*100:+.1f}%/yr")
    print(f"  Ramadan timing gross: {bt['strat_gross_ann']*100:+.1f}%/yr")
    print(f"  Ramadan timing net:   {bt['strat_net_ann']*100:+.1f}%/yr (30 bps one-way)")
    print(f"  Cost drag:            {bt['cost_drag_ann']*1e4:.0f} bps/yr\n")

    print("Verdict: Signal=WEAK (right sign + literature, but HAC t=1.55 < 2 on 11 months),")
    print("         Tradability=MIRAGE (net 2.2%/yr << buy-and-hold; no vehicle).")


if __name__ == "__main__":
    main()
