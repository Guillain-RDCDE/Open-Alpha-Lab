"""Real-panel verification -- Study 502 (Betting-Against-Correlation).

Reads the study's own yfinance cache (studies/502-betting-against-correlation/_cache/), runs
the Betting-Against-Correlation book (sort on trailing 252-day correlation-to-market, long the
low-correlation half, short the beta-neutralised high-correlation half, one execution lag),
and PRINTS every number that lands in docs/results.md: the leg cards, the gross/net headline
with HAC t and a block-bootstrap CI, the label-shuffle placebo p, the correlation-vs-beta-vs-vol
decomposition (the line that separates this from Study 238), and the seed-robust synthetic
positive control.

    python studies/502-betting-against-correlation/examples/verify.py

Pass --fetch to populate the cache from yfinance on first run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from betting_against_correlation import data, strategy as st  # noqa: E402


def main() -> None:
    fetch = "--fetch" in sys.argv
    prices, spy = data.fetch_panel(fetch=fetch)

    if prices.empty:
        print("Cache not found. Re-run with --fetch (network) from repo root to populate it.")
        return

    print("Study 502 -- Betting-Against-Correlation -- real yfinance panel\n")
    print(f"Prices panel: {prices.shape}, fingerprint={data.fingerprint(prices)}")
    print(f"SPY series:   {spy.shape}, fingerprint={data.fingerprint(spy)}")
    print(f"Date range:   {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    book = st.bac_portfolio(prices, spy, window=252, min_stocks=16)
    if book.empty:
        print("Could not construct the book (too few months).")
        return

    s = st.summary(book["bac_ret"])
    net = st.apply_costs(book, cost_bps=5.0, borrow_ann_bps=50.0)
    sn = st.summary(net)
    ci = st.block_bootstrap_ci(book["bac_ret"], seed=502)

    print("=== BAC book (long low-corr, short beta-neutralised high-corr) ===")
    print(f"  N months:            {s['n']} "
          f"({book.index[0].strftime('%Y-%m')} -- {book.index[-1].strftime('%Y-%m')})")
    print(f"  Gross mean (ann):    {s['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):           {s['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s['tstat']:+.3f}")
    print(f"  Hit rate:            {s['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s['max_drawdown']*100:.1f}%")
    print(f"  Bootstrap 95% CI:    [{ci[0]*100:+.1f}%, {ci[1]*100:+.1f}%] (ann mean)")
    print(f"  NET mean (5bps/leg + 50bps borrow): {sn['mean']*100:+.2f}%/yr  (t {sn['tstat']:+.3f})")

    print("\n=== Sort separation & neutralisation ===")
    print(f"  Avg correlation (long leg):  {book['avg_corr_long'].mean():.3f}")
    print(f"  Avg correlation (short leg): {book['avg_corr_short'].mean():.3f}")
    print(f"  Avg beta (long leg):         {book['avg_beta_long'].mean():.3f}")
    print(f"  Avg beta (short leg):        {book['avg_beta_short'].mean():.3f}")
    print(f"  Avg short-leg weight w_s:    {book['w_short'].mean():.3f} (beta-neutralising)")
    print(f"  Avg monthly turnover:        {book['turnover'].mean()*100:.1f}%")

    sl = st.summary(book["long_ret"]); ss = st.summary(book["short_ret"]); sm = st.summary(book["mkt_ret"])
    print(f"  Long leg (ann):  {sl['mean']*100:+.2f}%/yr | Short leg (ann): {ss['mean']*100:+.2f}%/yr | SPY: {sm['mean']*100:+.2f}%/yr")

    print("\n=== Label-shuffle placebo (correlation ranks permuted) ===")
    pb = st.placebo_pvalue(prices, spy, n_shuffles=200, seed=502, window=252, min_stocks=16)
    print(f"  Real mean / month:   {pb['real_mean']*100:+.4f}%")
    print(f"  Placebo mean / month: {pb['placebo_mean']*100:+.4f}% (+/- {pb['placebo_std']*100:.4f}%)")
    print(f"  One-sided p-value:   {pb['p_value']:.3f}  (n={pb['n_shuffles']} shuffles)")

    print("\n=== Decomposition -- is it correlation, not vol? (AFGP 2020) ===")
    for so, label in [("corr", "correlation (BAC)"), ("beta", "beta (BAB)"), ("vol", "volatility (BAV)")]:
        b = st.bac_portfolio(prices, spy, sort_on=so, window=252, min_stocks=16)
        d = st.summary(b["bac_ret"])
        print(f"  sort on {label:18s}: {d['mean']*100:+.2f}%/yr  Sharpe {d['sharpe']:+.3f}  t {d['tstat']:+.3f}")

    print("\n=== Year-by-year BAC returns ===")
    bc = book.copy(); bc["year"] = bc.index.year
    ann = bc.groupby("year")["bac_ret"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in ann.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\n=== Synthetic positive control (seed-robust, 20 seeds/point) ===")
    curve = st.synthetic_control_curve(n_seeds=20, base_seed=502)
    for prem, row in curve.iterrows():
        print(f"  planted premium {prem*100:5.1f}%/yr -> mean HAC t {row['mean_t']:+.3f} "
              f"(+/- {row['std_t']:.3f}, n={int(row['n_seeds'])})")

    print("\nNOTE Survivorship bias: universe = current large-cap names projected backwards (48).")
    print("     The high-correlation short leg is missing the failed names a real BAC short would")
    print("     have caught -> all positive results are UPPER-BOUND estimates. One execution lag")
    print("     (enter close 1 day after signal). Risk-free rate = 3%/yr constant (no FRED).")


if __name__ == "__main__":
    main()
