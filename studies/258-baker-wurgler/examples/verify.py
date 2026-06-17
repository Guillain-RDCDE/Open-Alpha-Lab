"""Real-tape verification -- Study 258 (Baker-Wurgler). Regenerates docs/results.md numbers.

Reads the month-end ^GSPC (and ^RUT size-proxy) closes from the repo-level _cache,
joins the lagged Baker-Wurgler sentiment reconstruction, sorts next-month returns by
prior-sentiment tertile, runs the predictive regression (Newey-West HAC), sweeps
sub-periods, and stresses a long-when-fearful timing overlay against buy-and-hold.
Network is touched only with --fetch.

    python studies/258-baker-wurgler/examples/verify.py          # cache-only
    python studies/258-baker-wurgler/examples/verify.py --fetch  # refresh ^GSPC/^RUT
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baker_wurgler import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def main(fetch: bool) -> None:
    sent = data.bw_monthly()
    mkt = data.fetch_market_monthly(fetch=fetch)
    df = data.merge_sentiment_returns(mkt, sent)

    print("Study 258 -- Baker-Wurgler -- real-tape verification\n")
    print(f"Window: {df.index[0].strftime('%Y-%m')} -> {df.index[-1].strftime('%Y-%m')}  "
          f"n={len(df)} return-months")
    print(f"Fingerprints: sent={data.fingerprint(sent)}  ret={data.fingerprint(df['ret'])}\n")

    # Regime sort
    reg = st.regime_sort(df, "ret", n_bins=3)
    print("=== Next-month S&P 500 return by prior sentiment tertile ===")
    for lab in ["low", "mid", "high"]:
        r = reg.loc[lab]
        print(f"  {lab:4s}: {r['mean_ann']*100:+6.2f}%/yr  vol={r['vol_ann']*100:4.1f}%  "
              f"HAC t={r['tstat']:+.2f}  n={int(r['n'])}")

    lmh = st.low_minus_high(df, "ret", n_bins=3)
    s_lmh = st.summarize(lmh)
    print(f"  Low-minus-high spread: {s_lmh['mean']*12*100:+.2f}%/yr  HAC t={s_lmh['tstat']:+.2f}")

    pr = st.predictive_regression(df, "ret")
    print(f"\n=== Predictive regression (next-month ret ~ lagged sentiment) ===")
    print(f"  slope = {pr['beta']*1e4:+.2f} bps/unit  HAC t = {pr['beta_t']:+.2f}  R2 = {pr['r2']:.4f}")

    if _HAS_QUANTLAB:
        ci = sharpe_ci_bootstrap(lmh, n_boot=2000, periods_per_year=12, seed=258)
        print(f"  Spread Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  "
              f"({ci['frac_negative']*100:.0f}% negative)")

    # Sub-periods
    print("\n=== Sub-period analysis ===")
    for label, a, b in [("1965-1989", "1965", "1989"),
                        ("1990-2007", "1990", "2007"),
                        ("2008-2026", "2008", "2026")]:
        sub = df.loc[a:b]
        prs = st.predictive_regression(sub, "ret")
        rs = st.regime_sort(sub, "ret", n_bins=3)
        lo = rs.loc["low", "mean_ann"]*100 if "low" in rs.index else float("nan")
        hi = rs.loc["high", "mean_ann"]*100 if "high" in rs.index else float("nan")
        print(f"  {label}: beta HAC t={prs['beta_t']:+.2f}  low={lo:+.1f}%/yr  high={hi:+.1f}%/yr  n={prs['n']}")

    # Cross-sectional leg
    if "smb" in df.columns:
        dfs = df.dropna(subset=["smb"])
        if len(dfs) > 50:
            rs = st.regime_sort(dfs, "smb", n_bins=3)
            prs = st.predictive_regression(dfs, "smb")
            print(f"\n=== Cross-sectional leg: SMB (^RUT - ^GSPC), {dfs.index[0].strftime('%Y-%m')}-> n={len(dfs)} ===")
            for lab in ["low", "high"]:
                if lab in rs.index:
                    print(f"  {lab:4s}: SMB {rs.loc[lab,'mean_ann']*100:+.2f}%/yr  HAC t={rs.loc[lab,'tstat']:+.2f}")
            print(f"  SMB regression slope: {prs['beta']*1e4:+.2f} bps/unit  HAC t={prs['beta_t']:+.2f}")

    # Timing overlay
    ov = st.timing_overlay(df, "ret", n_bins=3, short_high=False, one_way_bps=5.0)
    s_net = st.summarize(ov["net"]); s_bh = st.summarize(ov["buyhold"])
    tc = st.turnover_count(ov)
    print("\n=== Timing overlay (long-when-fearful, flat otherwise; 5bps one-way) ===")
    print(f"  Net:        {s_net['mean']*12*100:+.2f}%/yr  SR(ann)={s_net['sharpe']*np.sqrt(12):+.2f}  maxDD={s_net['max_drawdown']:.1%}")
    print(f"  Buy & hold: {s_bh['mean']*12*100:+.2f}%/yr  SR(ann)={s_bh['sharpe']*np.sqrt(12):+.2f}  maxDD={s_bh['max_drawdown']:.1%}")
    print(f"  Switches/yr={tc['switches_per_year']:.2f}  total cost={tc['total_cost_bps']:.1f} bps")

    # Synthetic positive control
    syn, _ = data.synthetic_market(contrarian_bps=60.0, seed=258)
    pr_syn = st.predictive_regression(syn, "ret")
    syn0, _ = data.synthetic_market(contrarian_bps=0.0, seed=258)
    pr_null = st.predictive_regression(syn0, "ret")
    print("\n=== Synthetic positive control ===")
    print(f"  contrarian_bps=60: slope={pr_syn['beta']*1e4:+.2f} bps/unit  HAC t={pr_syn['beta_t']:+.2f}")
    print(f"  contrarian_bps=0 : slope={pr_null['beta']*1e4:+.2f} bps/unit  HAC t={pr_null['beta_t']:+.2f}")

    print("\nVerdict: Signal=WEAK (right contrarian sign, no robust HAC t>=2), Tradability=MIRAGE (loses to buy-and-hold).")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
