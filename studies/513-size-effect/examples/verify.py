"""Real-panel verification -- Study 513 (Size-Effect / Banz 1981 SMB). Prints every number.

Reads the study's own yfinance cache (studies/513-size-effect/_cache/), ranks the survivor
basket by market cap each month, runs the small-minus-large long-short (dollar-neutral and
beta-neutral) with one execution lag, charges costs + short borrow, runs a label-shuffle
placebo, the January slice, the post-2000 decade slice, and a seed-robust synthetic control.

    python studies/513-size-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from size_effect import data, strategy as st  # noqa: E402


def main() -> None:
    prices, spy, caps = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run data.fetch_panel(fetch=True) from repo root with network.")
        return

    print("Study 513 -- Size-Effect (Banz 1981 SMB) -- real yfinance panel\n")
    print(f"Prices: {prices.shape}, fingerprint={data.fingerprint(prices)}")
    print(f"SPY:    {spy.shape}, fingerprint={data.fingerprint(spy)}")
    print(f"Caps:   {caps.shape}, fingerprint={data.fingerprint(caps)}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"Median cap split: large names = {(caps > caps.median()).sum()}, "
          f"small names = {(caps <= caps.median()).sum()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    book = st.smb_portfolio(prices, spy, caps, min_stocks=10)
    book = st.add_beta_neutral(book)
    if book.empty:
        print("Could not construct portfolio.")
        return

    book = data.drop_partial_last_month(book)

    s_small = st.summary(book["small_ret"])
    s_large = st.summary(book["large_ret"])
    s_mkt = st.summary(book["mkt_ret"])
    s_dn = st.summary(book["smb_dn"])
    s_bn = st.summary(book["smb_bn"])

    print("=== Legs (equal-weight, monthly) ===")
    print(f"  Small-cap leg:  {s_small['mean']*100:+.2f}%/yr  Sharpe {s_small['sharpe']:+.3f}")
    print(f"  Large-cap leg:  {s_large['mean']*100:+.2f}%/yr  Sharpe {s_large['sharpe']:+.3f}")
    print(f"  SPY (market):   {s_mkt['mean']*100:+.2f}%/yr  Sharpe {s_mkt['sharpe']:+.3f}")

    print("\n=== SMB long-short (small minus large), GROSS ===")
    print(f"  N months:            {s_dn['n']}  ({book.index[0].date()} -- {book.index[-1].date()})")
    print(f"  Dollar-neutral mean: {s_dn['mean']*100:+.2f}%/yr  Sharpe {s_dn['sharpe']:+.3f}")
    print(f"  Dollar-neutral HAC t:{s_dn['tstat']:+.3f}   hit {s_dn['hit_rate']*100:.1f}%  DD {s_dn['max_drawdown']*100:.1f}%")
    print(f"  Beta-neutral   mean: {s_bn['mean']*100:+.2f}%/yr  Sharpe {s_bn['sharpe']:+.3f}")
    print(f"  Beta-neutral   HAC t:{s_bn['tstat']:+.3f}")
    print(f"  Avg beta-hedge ratio:{book['beta_ratio'].mean():.3f}")

    # Costs
    net_dn = st.apply_costs(book["smb_dn"], book["turnover"], one_way_bps=10.0, borrow_annual_bps=100.0)
    s_net = st.summary(net_dn)
    print("\n=== SMB dollar-neutral, NET (10 bps/turnover + 100 bps/yr borrow) ===")
    print(f"  Avg monthly turnover:{book['turnover'].mean()*100:.1f}%")
    print(f"  Net mean:            {s_net['mean']*100:+.2f}%/yr  Sharpe {s_net['sharpe']:+.3f}")
    print(f"  Net HAC t:           {s_net['tstat']:+.3f}")

    # Placebo
    t_real, p = st.placebo_pvalue(prices, spy, caps, n_perm=300, min_stocks=10)
    print("\n=== Placebo (label-shuffle null, 300 permutations) ===")
    print(f"  Real DN HAC t:       {t_real:+.3f}")
    print(f"  Placebo p (|t_null| >= |t_real|): {p:.3f}")

    # January slice
    jan = st.january_split(book)
    print("\n=== January concentration ===")
    print(f"  January mean spread: {jan['jan_mean']*100:+.2f}%/mo  (n={jan['jan_n']})")
    print(f"  Other-months spread: {jan['rest_mean']*100:+.2f}%/mo  (n={jan['rest_n']})")
    print(f"  Welch t / p:         {jan['welch_t']:+.3f} / {jan['welch_p']:.3f}")

    # Decade slice (data begins 2001 -> early vs late SAMPLE split at the midpoint;
    # the classic Banz "post-1980 decay" read on a 2001-2026 survivor tape).
    dec = st.decade_split(book, cut="2013-07-01")
    print("\n=== Decay within sample (early 2001-2013 vs late 2013-2026) ===")
    print(f"  Early (2001-2013): {dec['pre_mean_ann']*100:+.2f}%/yr  HAC t {dec['pre_t']:+.3f}  (n={dec['pre_n']})")
    print(f"  Late  (2013-2026): {dec['post_mean_ann']*100:+.2f}%/yr  HAC t {dec['post_t']:+.3f}  (n={dec['post_n']})")

    # Synthetic control
    sc = st.synthetic_control(size_premium=0.06, n_seeds=25)
    print("\n=== Synthetic positive control (25 seeds, faithful-engine check) ===")
    print(f"  Mean HAC t with planted premium (6%/yr): {sc['mean_t_signal']:+.3f}  "
          f"(frac t>2 = {sc['frac_signal_t_gt2']:.2f})")
    print(f"  Mean HAC t under the null (0%):          {sc['mean_t_null']:+.3f}  "
          f"(frac |t|>2 = {sc['frac_null_t_gt2']:.2f})")

    print("\nNOTE Survivorship bias: basket = names still trading in 2026, projected backwards.")
    print("     Failed small-caps (the natural long) are absent -> results are upper bounds.")
    print("     Caps reconstructed as cap_now * price_t/price_now (shares-constant approximation).")


if __name__ == "__main__":
    main()
