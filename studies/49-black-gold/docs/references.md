# References & literature map — Study 49 (Black-Gold)

## The claim and its source

- **Driesprong, G., Jacobsen, B., & Maat, B. (2008).** *Striking Oil: Another Puzzle?* Journal of
  Financial Economics 89(2), 307–327 — the original: oil-price changes negatively predict next-month
  equity returns, in many markets, 1973–2003.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Crude Oil Predicts Equity Returns"* (listed Sharpe `0.599`), with a QuantConnect implementation.
  Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On out-of-sample failure of return predictors

- **Welch, I., & Goyal, A. (2008).** *A Comprehensive Look at the Empirical Performance of Equity
  Premium Prediction.* Review of Financial Studies 21(4) — most documented equity-return predictors
  fail out of sample; our negative result on oil is squarely in this tradition.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — post-publication decay.

## Data

- **Yahoo! Finance** — WTI crude front future (CL=F), the S&P 500 (^GSPC) and the 13-week T-bill
  (^IRX, the cash leg), 2000–2026, **daily** closes resampled to month-end. We deliberately avoid
  Yahoo's native monthly CL=F feed: it is full of holes (89 of 310 months missing on a 2026 pull),
  which silently mis-lags a positional one-month shift by 2–3 months; the daily-resampled grid is
  verified hole-free on every read. Honest caveat: CL=F begins in 2000, so we cannot test Driesprong's
  original 1973–2003 window on free data — but the strategy's value is its *out-of-sample* survival,
  and on all post-2000 (tradable) data the effect is absent. The offline synthetic world injects a
  tunable (negative-sign) oil→equity link and a null.

*The cross-asset entry on the debunk bench; companion in spirit to [47 Paper-Moon](../../47-paper-moon/)
(a predictor whose logic doesn't hold) and the decay studies.*
