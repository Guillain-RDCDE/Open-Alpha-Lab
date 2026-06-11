# References & literature map — Study 55 (Summer-Lull)

## The effect and its source

- **Bouman, S., & Jacobsen, B. (2002).** *The Halloween Indicator, "Sell in May and Go Away": Another
  Puzzle.* American Economic Review 92(5) — the foundational documentation across many markets.
- **Jacobsen, B., & Zhang, C. (2018).** *The Halloween Indicator: Everywhere and All the Time* — a
  300-year, 65-market follow-up confirming persistence (why our pattern holds in both halves).
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Turn of the Month / calendar"* family; the Halloween effect is the seasonal cousin of the calendar
  studies. Backlog: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why a real seasonal still doesn't justify the trade

- **Open-Alpha-Lab [Study 41 Hangover](../../41-hangover/) and [Study 42 Last-Call](../../42-last-call/)**
  — the exposure-reduction illusion: a rule that sits out a positive-return window lowers wealth at the
  same Sharpe. The same logic kills "sell in May" even though the seasonal is real.

## Data

- **Yahoo! Finance** — S&P 500 (^GSPC) daily auto-adjusted close resampled to month-end total return,
  1928–2026. The offline synthetic world injects a known winter (Nov–Apr) premium (and a null) to
  exercise the machinery without the network.

*The calendar bench: [41 Hangover](../../41-hangover/) (January Barometer) · [42 Last-Call](../../42-last-call/)
(turn of the month) · [48 Groundhog](../../48-groundhog/) (the seasonal that *is* tradable).*
