"""Reproducible headline run for Study 516 — Dividend-Month-Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices + dividend dates
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dividend_month_premium import data, strategy as st

NDRAWS = 20_000

print("# Dividend-Month-Premium — fixed 30-name large-cap survivor basket (yfinance)")
if data.have_real():
    prices, divs = data.load_real()
    flags = data.build_predicted_flags(prices, divs, min_history=data.MIN_HISTORY)
    span = (prices.index.max() - prices.index.min()).days / 365.25
    n_names = len(flags)
    frac = st.predicted_month_fraction(flags)
    print(f"prices         : {prices.shape[0]} days x {prices.shape[1]} names, "
          f"{prices.index.min().date()} -> {prices.index.max().date()} ({span:.1f} years)")
    print(f"dividends       : {len(divs)} payment dates across {divs['ticker'].nunique()} names")
    print(f"predicted-month%: {frac*100:.2f}% of all stock-months are predicted dividend months "
          f"({n_names} names with usable monthly history)")

    print("\n# Per-month premium — predicted dividend months vs the rest")
    s = st.summarize_premium(flags, n_draws=NDRAWS)
    print(f"  predicted mean: {s['pred_mean_bps']:+.2f} bps/month  (n={s['n_pred']:,})")
    print(f"  rest mean     : {s['rest_mean_bps']:+.2f} bps/month  (n={s['n_rest']:,})")
    print(f"  premium       : {s['premium_bps']:+.2f} bps/month   Welch t = {s['welch_t']:+.2f}")
    print(f"  per-NAME prem : {s['name_premium_bps']:+.2f} bps/month  one-sample t = "
          f"{s['t_name']:+.2f}  (n_names={s['n_names']})")
    print(f"  placebo p     : {s['p_placebo']:.4f}  (random same-density calendar tags)")

    print("\n# Share-of-return decomposition")
    sr = st.share_of_return(flags)
    print(f"  {sr['share_months']*100:.2f}% of months are predicted dividend months; they carry "
          f"{sr['share_return']*100:.2f}% of cumulative log return")
    print(f"  concentration ratio = {sr['share_return']/sr['share_months']:.2f}x "
          f"(share of return / share of months)")

    print("\n# Tradable overlay — long the basket ONLY in predicted dividend months")
    for cb in (2.0, 5.0, 10.0):
        ov = st.overlay_stats(flags, cost_bps=cb)
        print(f"  cost={cb:>4.1f} bps/leg: gross {ov['gross_monthly_bps']:+.2f} bps/mo -> "
              f"net {ov['net_monthly_bps']:+.2f} bps/mo | buy&hold {ov['bh_monthly_bps']:+.2f} bps/mo"
              f"  (net ann {ov['net_ann_pct']:+.1f}% vs B&H {ov['bh_ann_pct']:+.1f}%)")

    print("\n# Third axis — is the premium really PREDICTABLE in advance? (vs the actual ex-div month)")
    fa = st.build_actual_flags(prices, divs)
    sa = st.summarize_premium(fa, n_draws=NDRAWS)
    print(f"  PREDICTED (past-only) : prem {s['premium_bps']:+.2f} bps/mo  Welch t={s['welch_t']:+.2f}  "
          f"per-name t={s['t_name']:+.2f}  p={s['p_placebo']:.4f}")
    print(f"  ACTUAL ex-div month   : prem {sa['premium_bps']:+.2f} bps/mo  Welch t={sa['welch_t']:+.2f}  "
          f"per-name t={sa['t_name']:+.2f}  p={sa['p_placebo']:.4f}")

    print("\n# Robustness — min payment-history required to call a month 'predicted'")
    for mh in (1, 2, 4):
        fl = data.build_predicted_flags(prices, divs, min_history=mh)
        s2 = st.summarize_premium(fl, placebo=False)
        print(f"  min_history={mh}: premium {s2['premium_bps']:+.2f} bps/mo  "
              f"Welch t={s2['welch_t']:+.2f}  per-name t={s2['t_name']:+.2f}  "
              f"(predicted-month share {st.predicted_month_fraction(fl)*100:.1f}%)")
else:
    print("(no _cache/dmp_prices.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the predicted-vs-rest detector must recover a PLANTED premium and must NOT manufacture")
print("  significance when the true premium is 0.")
for edge in (0.0, 0.010):
    px, dv = data.synthetic_panel(edge=edge, seed=516)
    fl = data.build_predicted_flags(px, dv, min_history=data.MIN_HISTORY)
    s = st.summarize_premium(fl, n_draws=4_000)
    print(f"  planted edge={edge*1e4:+7.1f} bps/mo: premium={s['premium_bps']:+.2f} bps/mo  "
          f"Welch t={s['welch_t']:+6.2f}  per-name t={s['t_name']:+6.2f}  "
          f"placebo p={s['p_placebo']:.3f}")
