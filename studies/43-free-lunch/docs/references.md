# References & literature map — Study 43 (Free-Lunch)

## The strategy and its source

- **Frazzini, A., & Pedersen, L. H. (2014).** *Betting Against Beta.* Journal of Financial Economics
  111(1), 1–25 — the BAB factor: long low-beta (levered to 1), short high-beta, the leverage-constraint
  thesis.
- **Black, F. (1972).** *Capital Market Equilibrium with Restricted Borrowing.* Journal of Business
  45(3) — the original flat security-market-line observation BAB monetises.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Betting Against Beta Factor in Stocks"* (listed Sharpe `0.594`), with a QuantConnect implementation.
  Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## The leverage critique

- **Novy-Marx, R., & Velikov, M. (2022).** *Betting Against Betting Against Beta.* Journal of Financial
  Economics — BAB's performance is sensitive to construction and the (often unmodelled) leverage and
  transaction assumptions; our financing sweep is in that spirit.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — the post-publication-decay frame.

## Data

- **Yahoo! Finance** — 13 liquid ETFs spanning the beta spectrum + SPY as the market, daily
  auto-adjusted, 2000–2026. ETFs (not single stocks) keep the run fully tradable and survivorship-free
  at the cost of coarser beta dispersion — a trade-off stated openly in the results. The offline
  synthetic factor world injects known betas and a tunable low-beta premium (and a null).

*Sibling: [30 House-Edge](../../30-house-edge/) — the same lesson, that "beats the market after free
leverage" isn't beating the market; [40 Paper-Tiger](../../40-paper-tiger/), [41 Hangover](../../41-hangover/),
[42 Last-Call](../../42-last-call/) — the benchmark-illusion bench.*
