# References & literature map — Study 50 (High-Water)

## The effect and its source

- **George, T., & Hwang, C.-Y. (2004).** *The 52-Week High and Momentum Investing.* Journal of Finance
  59(5), 2145–2176 — nearness to the 52-week high predicts returns, argued to dominate and be distinct
  from Jegadeesh-Titman momentum (an anchoring story).
- **Jegadeesh, N., & Titman, J. (1993).** *Returns to Buying Winners and Selling Losers* — the momentum
  benchmark we correlate against.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"52-Weeks High Effect in Stocks"* (listed Sharpe `0.153`), with a QuantConnect implementation.
  Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On momentum crashes and the large-cap reversal

- **Daniel, K., & Moskowitz, T. (2016).** *Momentum Crashes.* Journal of Financial Economics — near-high
  / winner portfolios crash hard in rebounds (the dot-com and GFC episodes that sink the sign here).
- **Open-Alpha-Lab [Study 33 Slingshot](../../33-slingshot/)** — short-term reversal is real on the S&P
  500 cross-section; the negative 52-week-high sign on large-cap survivors is the same coin.

## Data

- **Yahoo! Finance** — monthly total returns for current S&P 500 members with ≥20 years of history.
  Honest caveats: survivorship-biased and large-cap (the effect is documented across the size spectrum;
  the large-cap survivor universe favours reversal). The offline synthetic trending panel makes nearness
  and momentum both predictive and correlated (and a null), so the "same factor" result is provable offline.

*The relabelled-factor entry on the bench; mechanistically tied to
[33 Slingshot](../../33-slingshot/) (reversal) and [24 Stampede](../../24-stampede/) (momentum).*
