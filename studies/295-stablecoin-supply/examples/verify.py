"""Real-tape verification -- Study 295 (Stablecoin-Supply). Regenerates docs/results.md numbers.

Joins the curated monthly total-stablecoin-supply series with the (cache-only)
BTC-USD monthly tape, builds the supply-growth timing signal (dry-powder thesis),
and pins it against buy-and-hold BTC: a predictive regression (lagged vs
contemporaneous = the reflexivity check), sub-period decay, turnover/cost sweep,
and a bootstrap CI on the timed-minus-buy-and-hold excess.
Network is touched only with --fetch.

    python studies/295-stablecoin-supply/examples/verify.py          # cache-only
    python studies/295-stablecoin-supply/examples/verify.py --fetch  # refresh the BTC tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stablecoin_supply import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def main(fetch: bool) -> None:
    btc = data.fetch_btc_monthly(fetch=fetch)
    panel = data.build_real_panel(btc)
    print(f"Panel: {len(panel)} months  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"BTC fingerprint: {data.fingerprint(btc)}")
    print(f"Panel fingerprint: {data.fingerprint(panel)}\n")

    rets = st.timed_returns(panel, window=1, threshold=0.0, one_way_bps=0.0)
    s_btc = st.summarize(rets["btc"])
    s_timed = st.summarize(rets["timed"])
    excess = (rets["timed"] - rets["btc"]).dropna()
    s_exc = st.summarize(excess)

    print("=== Supply-timed (long when supply grew last month) vs buy-and-hold BTC ===")
    print(f"Months: {len(rets)}  (long {int(rets['weight'].sum())} / flat {int((rets['weight']==0).sum())})")
    print(f"BTC   : {s_btc['mean']*12*100:+.1f}%/yr  SR(ann)={s_btc['sharpe']*np.sqrt(12):+.2f}  HAC t={s_btc['tstat']:+.2f}  maxDD={s_btc['max_drawdown']*100:.0f}%")
    print(f"TIMED : {s_timed['mean']*12*100:+.1f}%/yr  SR(ann)={s_timed['sharpe']*np.sqrt(12):+.2f}  HAC t={s_timed['tstat']:+.2f}  maxDD={s_timed['max_drawdown']*100:.0f}%")
    print(f"EXCESS (timed - BTC): {s_exc['mean']*12*100:+.1f}%/yr  HAC t={s_exc['tstat']:+.2f}  hit={s_exc['hit_rate']:.3f}")

    if _HAS_QUANTLAB:
        ci = sharpe_ci_bootstrap(excess, n_boot=2000, periods_per_year=12, seed=295)
        print(f"Excess Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% negative)")

    print("\n=== Predictive regression: BTC ret ~ standardised supply growth ===")
    r1 = st.predictive_regression(panel, window=1, lag=1)
    r0 = st.predictive_regression(panel, window=1, lag=0)
    print(f"LAGGED (t+1, tradable): beta={r1['beta']:+.4f}  HAC t={r1['tstat']:+.2f}  R2={r1['r2']:.4f}  corr={r1['corr']:+.3f}  n={r1['n']}")
    print(f"CONTEMPORANEOUS (t, NOT tradable -- reflexivity): beta={r0['beta']:+.4f}  HAC t={r0['tstat']:+.2f}  R2={r0['r2']:.4f}  corr={r0['corr']:+.3f}  n={r0['n']}")
    print("  -> contemporaneous > lagged is the signature of a coincident, reflexive series.")

    print("\n=== Sub-period decay (lagged regression) ===")
    for label, a, b in [("2018-2020", "2018", "2020"), ("2021-2022", "2021", "2022"), ("2023-2026", "2023", "2026")]:
        r = st.predictive_regression(panel[a:b], window=1, lag=1)
        print(f"  {label}: beta={r['beta']:+.4f}  HAC t={r['tstat']:+.2f}  n={r['n']}")

    print("\n=== Turnover and cost drag ===")
    print(f"Avg monthly switch rate: {st.turnover(panel):.2%}")
    for bps in (10, 25, 50):
        rn = st.timed_returns(panel, one_way_bps=bps)
        sn = st.summarize(rn["timed_net"])
        print(f"  Net timed @{bps}bps one-way: {sn['mean']*12*100:+.1f}%/yr  HAC t={sn['tstat']:+.2f}")

    print("\n=== Positive control (synthetic, planted dry-powder premium) ===")
    for prem in (0.0, 0.04):
        dfp, _ = data.synthetic_panel(premium=prem, seed=295)
        rp = st.predictive_regression(dfp, window=1, lag=1)
        print(f"  premium={prem:.2f}: lagged beta={rp['beta']:+.4f}  HAC t={rp['tstat']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
