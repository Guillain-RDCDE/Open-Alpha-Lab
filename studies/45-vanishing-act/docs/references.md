# References & literature map — Study 45 (Vanishing-Act)

## The effect and its source

- **Banz, R. (1981).** *The Relationship Between Return and Market Value of Common Stocks.* Journal of
  Financial Economics 9(1) — the original size effect.
- **Fama, E., & French, K. (1993).** *Common Risk Factors in the Returns on Stocks and Bonds.* Journal
  of Financial Economics 33(1) — SMB as a pricing factor.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Size Factor – Small Capitalization Stocks Premium"* (listed Sharpe `0.747`), with a QuantConnect
  implementation. Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## The decay and the quality caveat

- **van Dijk, M. (2011).** *Is Size Dead? A Review of the Size Effect in Equity Returns.* Journal of
  Banking & Finance 35(12) — documents the post-1981 disappearance.
- **Asness, C., Frazzini, A., Israel, R., Moskowitz, T., & Pedersen, L. (2018).** *Size Matters, If You
  Control Your Junk.* Journal of Financial Economics — the surviving size effect is really a quality
  story (why our S&P 600 / IJR pair, which screens for profitability, does marginally best).
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — the post-publication-decay frame this study exemplifies.

## Data

- **Yahoo! Finance** — ^RUT/^GSPC (1987–2026, the long history) and the total-return ETF pairs
  IWM/SPY and IJR/IVV (2000–2026). The offline synthetic world injects a size premium that can ramp
  from positive to negative (and a null) to exercise the machinery without the network.

*The decay bench: [40 Paper-Tiger](../../40-paper-tiger/), [41 Hangover](../../41-hangover/),
[42 Last-Call](../../42-last-call/), [43 Free-Lunch](../../43-free-lunch/),
[44 Growth-Spurt](../../44-growth-spurt/) — five ways a headline dies.*
