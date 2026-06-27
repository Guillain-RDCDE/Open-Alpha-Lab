"""Real-panel verification -- Study 514 (Pastor-Stambaugh Liquidity Risk).

Reads the study's own yfinance cache (studies/514-pastor-stambaugh-liquidity/_cache/),
builds the Amihud-style aggregate liquidity series, estimates each name's liquidity
beta, sorts high-minus-low gamma into a monthly long-short, and PRINTS every number that
appears in docs/results.md (data stamp, signal table, costs, robustness, placebo,
synthetic control).

    python studies/514-pastor-stambaugh-liquidity/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pastor_stambaugh_liquidity import data, strategy as st  # noqa: E402


def main() -> None:
    close, dv, spy = data.fetch_panel()
    if close.empty:
        print("Cache not found. Run from repo root with network access to populate the cache.")
        return

    print("Study 514 -- Pastor-Stambaugh Liquidity Risk -- real yfinance panel\n")
    print(f"Close panel: {close.shape}, fingerprint={data.fingerprint(close)}")
    print(f"Dollar-vol:  {dv.shape}, fingerprint={data.fingerprint(dv)}")
    print(f"SPY series:  {spy.shape}, fingerprint={data.fingerprint(spy)}")
    print(f"Date range:  {close.index[0].date()} -- {close.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted close + volume\n")

    # Aggregate liquidity
    liq = st.aggregate_liquidity(close, dv)
    shock = st.liquidity_shock(liq)
    print(f"Aggregate-liquidity months: {len(liq)} | shock months: {len(shock)}")
    print(f"Liquidity shock std: {shock.std():.4f}\n")

    # Long-short sort (gross), with one execution lag baked in
    res_gross = st.liquidity_sort(close, dv, spy, window=st.BETA_WINDOW_M)
    s_g = st.summary(res_gross["ls_gross"])
    s_long = st.summary(res_gross["long_ret"])
    s_short = st.summary(res_gross["short_ret"])

    print("=== Long high-gamma / short low-gamma (GROSS) ===")
    print(f"  N months:            {s_g['n']}")
    print(f"  Period:              {res_gross.index[0].date()} -- {res_gross.index[-1].date()}")
    print(f"  Mean return (ann):   {s_g['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):           {s_g['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s_g['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s_g['tstat']:+.3f}")
    print(f"  Hit rate:            {s_g['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s_g['max_drawdown']*100:.1f}%")
    print(f"  Long leg (ann):      {s_long['mean']*100:+.2f}%/yr")
    print(f"  Short leg (ann):     {s_short['mean']*100:+.2f}%/yr")
    print(f"  Avg turnover:        {res_gross['turnover'].mean():.2f} (one-way, both legs)")

    # Net of costs + borrow
    res_net = st.liquidity_sort(
        close, dv, spy, window=st.BETA_WINDOW_M,
        cost_bps=st.COST_BPS, borrow_ann_bps=st.BORROW_ANN_BPS,
    )
    s_n = st.summary(res_net["ls_net"])
    print("\n=== Net of costs (10 bps one-way x turnover) + 50 bps/yr borrow on short ===")
    print(f"  Mean return (ann):   {s_n['mean']*100:+.2f}%/yr")
    print(f"  Sharpe (ann):        {s_n['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s_n['tstat']:+.3f}")

    # Placebo / label-shuffle null
    print("\n=== Placebo: shuffle liquidity-beta labels, re-sort (200 perms) ===")
    pb = st.placebo_pvalue(close, dv, spy, window=st.BETA_WINDOW_M, n_perm=200)
    print(f"  Real long-short mean (ann): {pb['real_mean_ann']*100:+.2f}%/yr")
    print(f"  Placebo mean (ann):         {pb['placebo_mean']*100:+.2f}%/yr")
    print(f"  Placebo std (ann):          {pb['placebo_std']*100:.2f}%/yr")
    print(f"  Two-sided p-value:          {pb['p_value']:.3f}")

    # Robustness: window sweep
    print("\n=== Robustness: liquidity-beta window sweep ===")
    for w in (24, 36, 48):
        r_w = st.liquidity_sort(close, dv, spy, window=w)
        if r_w.empty:
            print(f"  window={w}m: (too few months)")
            continue
        s_w = st.summary(r_w["ls_gross"])
        print(f"  window={w}m: mean {s_w['mean']*100:+.2f}%/yr | t={s_w['tstat']:+.3f} | n={s_w['n']}")

    # Robustness: tertile vs quintile
    print("\n=== Robustness: sort breadth (tertile vs quintile) ===")
    for q, lbl in ((1.0/3.0, "tertile"), (0.2, "quintile")):
        r_q = st.liquidity_sort(close, dv, spy, window=st.BETA_WINDOW_M, quantile=q)
        if r_q.empty:
            continue
        s_q = st.summary(r_q["ls_gross"])
        print(f"  {lbl}: mean {s_q['mean']*100:+.2f}%/yr | t={s_q['tstat']:+.3f} | n={s_q['n']}")

    # Synthetic positive control
    print("\n=== Synthetic positive control (engine faithfulness; NOT a real-tape claim) ===")
    ctrl = st.control_sweep([-0.08, 0.0, 0.06, 0.12], window=30)
    for _, row in ctrl.iterrows():
        print(f"  planted premium {row['premium']:+.2f}: "
              f"ls mean {row['ls_mean_ann']*100:+.2f}%/yr | t={row['tstat']:+.3f}")

    print("\nNOTE Survivorship bias: universe = current large-cap names projected backwards.")
    print("     Failed/illiquid names (the natural source of liquidity risk) are absent;")
    print("     results are upper-bound estimates. RF approximated at 3%/yr (constant).")


if __name__ == "__main__":
    main()
