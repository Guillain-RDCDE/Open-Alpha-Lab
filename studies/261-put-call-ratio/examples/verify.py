"""Real-tape verification -- Study 261 (Put-Call-Ratio). Regenerates docs/results.md numbers.

Reads (or fetches) the monthly ^GSPC series, joins it with the hardcoded CBOE total
put/call ratio, and tests the contrarian claim: does an extreme-high put/call reading
time market bottoms? Reports the predictive regression, the extreme-fear timing rule
vs buy-and-hold (gross and net), the conditional next-month return, a random-timing
null, and sub-periods. Network is touched only with --fetch.

    python studies/261-put-call-ratio/examples/verify.py          # cache-only
    python studies/261-put-call-ratio/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from put_call_ratio import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.build_real_panel(fetch=fetch)
    print(f"Panel: {len(panel)} months  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Put/call mean={panel['pcr'].mean():.3f}  std={panel['pcr'].std():.3f}")
    print(f"Fingerprint: {data.fingerprint(panel)}\n")

    # ---- Continuous predictive regression -------------------------------
    reg = st.predictive_regression(panel)
    print("=== Next-month S&P return on expanding put/call z-score (HAC) ===")
    print(f"  slope={reg['slope']:+.4f}  HAC t={reg['tstat']:+.2f}  R2={reg['r2']:.4f}  n={reg['n']}")
    print("  (a contrarian edge => POSITIVE slope)\n")

    # ---- Extreme-fear timing rule vs buy-and-hold -----------------------
    tr = st.timing_returns(panel, q=0.80, one_way_bps=5.0)
    s_strat = st.summarize(tr["strat"])
    s_net = st.summarize(tr["strat_net"])
    s_bh = st.summarize(tr["bh"])
    n_switch = int(tr["in_mkt"].diff().abs().sum())
    print("=== Extreme-fear timing (q=0.80) vs buy-and-hold ===")
    print(f"Active months: {len(tr)}  in-market: {tr['in_mkt'].mean():.1%}  switches: {n_switch}")
    print(f"STRAT gross : {s_strat['mean']*1200:+.2f}%/yr  t={s_strat['tstat']:+.2f}  SR(ann)={s_strat['sharpe']*np.sqrt(12):+.2f}  hit={s_strat['hit_rate']:.3f}  maxDD={s_strat['max_drawdown']:.1%}")
    print(f"STRAT net   : {s_net['mean']*1200:+.2f}%/yr  t={s_net['tstat']:+.2f}")
    print(f"BUY & HOLD  : {s_bh['mean']*1200:+.2f}%/yr  t={s_bh['tstat']:+.2f}  SR(ann)={s_bh['sharpe']*np.sqrt(12):+.2f}  maxDD={s_bh['max_drawdown']:.1%}\n")

    # ---- Conditional next-month return ----------------------------------
    sig = st.extreme_signal(panel["pcr"], q=0.80).iloc[36:]
    fwd = panel["fwd_ret"].iloc[36:]
    fear = fwd[sig == 1]
    calm = fwd[sig == 0]
    try:
        from scipy import stats as sc
        wt, wp = sc.ttest_ind(fear, calm, equal_var=False)
    except Exception:
        wt, wp = float("nan"), float("nan")
    print("=== Conditional next-month return ===")
    print(f"After extreme fear: {fear.mean()*100:+.3f}%/mo  n={len(fear)}")
    print(f"After calm        : {calm.mean()*100:+.3f}%/mo  n={len(calm)}")
    print(f"Diff (fear-calm)  : {(fear.mean()-calm.mean())*100:+.3f}%/mo  Welch t={wt:+.2f}  p={wp:.3f}\n")

    # ---- Random-timing null --------------------------------------------
    rand = st.random_timing_returns(panel, in_mkt_frac=tr["in_mkt"].mean(), n_draws=2000, seed=261)
    perm_p = float((rand >= fear.mean()).mean())
    print(f"Random-timing null: mean={rand.mean()*100:+.3f}%/mo  perm p(random>=fear)={perm_p:.3f}\n")

    # ---- Sub-periods ----------------------------------------------------
    print("=== Sub-periods (extreme-fear timing vs buy-and-hold) ===")
    for lab, a, b in [("2006-2012", "2006", "2012"), ("2013-2019", "2013", "2019"), ("2020-2025", "2020", "2025")]:
        sub = tr["strat"][a:b]
        sbh = tr["bh"][a:b]
        s = st.summarize(sub)
        sb = st.summarize(sbh)
        print(f"  {lab}: strat {s['mean']*1200:+.1f}%/yr t={s['tstat']:+.2f} n={s['n']} | buy-hold {sb['mean']*1200:+.1f}%/yr")

    # ---- Synthetic positive control ------------------------------------
    print("\n=== Synthetic positive control (predictive regression) ===")
    for beta in (0.0, 0.15, 0.30):
        dfx, _ = data.synthetic_panel(contrarian_beta=beta, seed=261)
        r = st.predictive_regression(dfx)
        print(f"  beta={beta:<4}: slope={r['slope']:+.4f}  HAC t={r['tstat']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
