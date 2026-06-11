# References & literature map — Study 42 (Last-Call)

## The effect and its source

- **Lakonishok, J., & Smidt, S. (1988).** *Are Seasonal Anomalies Real? A Ninety-Year Perspective.*
  Review of Financial Studies 1(4) — the foundational turn-of-the-month documentation.
- **McConnell, J., & Xu, W. (2008).** *Equity Returns at the Turn of the Month.* Financial Analysts
  Journal 64(2) — the modern restatement (returns accrue almost entirely in the [-1,+3] window).
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Turn of the Month in Equity Indexes"* (listed Sharpe `0.305`), with a QuantConnect implementation.
  Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why "real per-day" ≠ "tradable"

- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — the post-publication-decay frame behind the sub-period split.

## Data

- **Yahoo! Finance** — S&P 500 (^GSPC) daily auto-adjusted close, 1950–2026 (for the effect), and SPY
  daily, 1993–2026 (for the tradable/cost run). The offline synthetic daily world injects a known
  turn-of-the-month bump (and a null) to exercise the machinery without the network.

*Sibling: [41 Hangover](../../41-hangover/) — the same exposure-reduction illusion in a calendar omen;
[40 Paper-Tiger](../../40-paper-tiger/) — another "the per-period number looks great" that fails the
right benchmark.*
