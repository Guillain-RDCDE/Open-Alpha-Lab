# References & literature map — Study 604 (Month-End Rebalancing Flows)

## The claim under test

- **The formal version.** Erkko Etula, Kalle Rinne, Matti Suominen & Lauri Vaittinen, *Dash for
  Cash: Monthly Market Impact of Institutional Liquidity Needs* (2020, **Review of Financial
  Studies** 33(1), 75–111; earlier drafts circulated as a Journal-of-Finance-track paper).
  They document systematic month-turn return patterns driven by **institutional liquidity and
  rebalancing cycles**: pressure into month-end, reversal in the first days — the exact
  two-legged shape we test. <https://academic.oup.com/rfs/article/33/1/75/5488008>
- **Who rebalances mechanically.** Jonathan A. Parker, Antoinette Schoar & Yang Sun, *Retail
  Financial Innovation and Stock Market Dynamics: The Case of Target Date Funds* (2023,
  **Journal of Finance** 78(5)). Target-date and balanced funds trade **contrarian to the
  month's equity-bond gap** by construction — after stocks trounce bonds they must sell
  equities — and the paper measures the market impact of those flows.
  <https://onlinelibrary.wiley.com/doi/10.1111/jofi.13258>
- **The practitioner folklore.** "Pension rebalancing estimates" are a fixture of sell-side
  month-end notes (Goldman/BAML/Wells "$X bn to sell" headlines). The claim's operational form —
  *big month-to-date equity-bond gap ⇒ equity sold into the close of the month, bounce after* —
  is what we test directly.

## The unconditional cousin (and the dedup guard)

- **Turn-of-the-month.** Robert A. Ariel, *A Monthly Effect in Stock Returns* (1987, JFE);
  Josef Lakonishok & Seymour Smidt, *Are Seasonal Anomalies Real? A Ninety-Year Perspective*
  (1988, RFS); John J. McConnell & Wei Xu, *Equity Returns at the Turn of the Month* (2008,
  FAJ). The desk already tested the **unconditional** calendar drift as
  [study 89-turn-of-the-month](../../89-turn-of-the-month/) — *this* study is the
  **conditional, flow-driven** claim: the two are distinct hypotheses, and our third axis tests
  the distinction explicitly (does TOM vanish when there is no gap to rebalance? — it does not).

## Method citations

- **Welch t** for the quintile splits: B. L. Welch (1947), *The generalization of "Student's"
  problem when several different population variances are involved*, Biometrika 34.
- **Newey-West (HAC) t** for the daily tradability stream: Whitney K. Newey & Kenneth D. West
  (1987), *A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*, Econometrica 55.
- **Permutation placebo** (gap shuffled across months, averaged over ≥ 20 seeds): Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap* (1993).
- **Costs discipline.** Andrea Frazzini, Ronen Israel & Tobias J. Moskowitz (2018), *Trading
  Costs*, SSRN 3229719 — the gross-vs-net gap that kills paper anomalies.
- **Post-publication decay.** R. David McLean & Jeffrey Pontiff (2016), *Does Academic Research
  Destroy Stock Return Predictability?*, JF 71 — the 2004–2015-then-gone pattern we find is the
  textbook decay footprint.

## Data sources

- **yfinance** total-return (auto-adjusted) closes: **SPY** (equity leg, 1993-01-29 →),
  **AGG** (iShares Core U.S. Aggregate Bond ETF, 2003-09-29 →) and **VBMFX** (Vanguard Total
  Bond Market Index Fund, same Bloomberg Aggregate index, used before the AGG inception).
  Cached under [`_cache/merf_prices.csv`](../_cache/merf_prices.csv). The **splice is in return
  space** at 2003-09-30 and is documented in [`data.py`](../month_end_rebalancing_flows/data.py)
  and [results.md](results.md). The plan's TLT alternative was rejected: TLT starts 2002-07 and
  its 20+year duration is a poor proxy for the Aggregate bond leg institutions rebalance against.

## Method lineage (the desk's shared engine)

- **Month table (gap conditioning + windows).**
  [`strategy.month_table`](../month_end_rebalancing_flows/strategy.py) — `gap_pre` uses only
  closes printed before the last-3 window; `gap_full` only closes printed before the first-3
  window. One execution lag everywhere.
- **Conditional split + dose-response.** [`strategy.quintile_split`](../month_end_rebalancing_flows/strategy.py),
  [`strategy.quintile_profile`](../month_end_rebalancing_flows/strategy.py).
- **Seed-averaged permutation placebo.** [`strategy.placebo_pvalue`](../month_end_rebalancing_flows/strategy.py)
  — 20 seeds × 2,000 draws.
- **Expanding-quintile flow trade.** [`strategy.flow_trade`](../month_end_rebalancing_flows/strategy.py)
  — no look-ahead thresholds, 8 one-way tickets per event, borrow on the short leg, HAC t.
- **Deterministic synthetic control.** [`data.synthetic_world`](../month_end_rebalancing_flows/data.py)
  — plants a tunable gap-conditional reversal; the null must stay quiet (verified over 20 seeds).

## Related desk studies

- [89-turn-of-the-month](../../89-turn-of-the-month/) — the **unconditional** calendar drift;
  our third axis shows it is *not* this flow effect in disguise.
- [97-balancing-act](../../97-balancing-act/) — rebalancing as a portfolio policy (the payer of
  these flows, not the trader of them).
- [602-macro-announcement-premium](../../602-macro-announcement-premium/) and
  [603-treasury-auction-concession](../../603-treasury-auction-concession/) — the same
  scheduled-flow / calendar-pressure family on other tapes.
