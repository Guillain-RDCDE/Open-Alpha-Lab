# References & literature map — Study 54 (Static)

## The effect and its source

- **Ang, A., Hodrick, R., Xing, Y., & Zhang, X. (2006).** *The Cross-Section of Volatility and Expected
  Returns.* Journal of Finance 61(1) — the idiosyncratic-volatility puzzle: high idio-vol precedes low
  returns, contrary to risk-return intuition.
- **Bali, T., & Cakici, N. (2008).** *Idiosyncratic Volatility and the Cross-Section of Expected
  Returns* — the puzzle is sensitive to universe, weighting and breakpoints; concentrated in small caps.
- **Vendor / literature family** — the low-risk/low-vol anomaly cluster; backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why it inverts on tradable large caps

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies* — idio-vol/lottery effects
  concentrate in micro-caps and weaken (or vanish) with value-weighting and NYSE breakpoints.
- **Open-Alpha-Lab** twins: **[53 Jackpot](../../53-jackpot/)** (MAX — the same high-risk axis) and
  **[43 Free-Lunch](../../43-free-lunch/)** (betting against beta, also failed on large caps).

## Data

- **Yahoo! Finance** — daily returns for current S&P 500 members + SPY (the market factor), ≥15y history.
  Survivorship-biased, large-cap (the puzzle's home is small/illiquid names). The offline synthetic panel
  makes high-idio-vol stocks underperform (and a null) so the machinery is provable offline.

*The near-twin of [53 Jackpot](../../53-jackpot/): MAX and idio-vol measure the same high-risk axis and
invert together on large caps.*
