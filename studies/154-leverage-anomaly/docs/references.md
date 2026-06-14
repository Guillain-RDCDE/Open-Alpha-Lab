# References & literature map — Study 154 (Leverage-Anomaly)

## The claim under test

- **Penman, Richardson & Tuna (2007).** *The Book-to-Price Effect in Stock Returns and
  Accounting for Leverage.* Journal of Accounting Research, 45(2), 427–467.
  The canonical source: they decompose the book-to-price ratio into an *operating* component
  and a *financial-leverage* component and find that the leverage component is negatively
  priced — high financial leverage predicts *lower* future returns, contrary to MM risk
  compensation.  This is the effect we attempt to replicate using the two simplest proxies
  from their decomposition (LTD/Assets and the D/E ratio).

- **Modigliani, F. & Miller, M. (1958).** *The Cost of Capital, Corporation Finance and the
  Theory of Investment.* American Economic Review, 48(3), 261–297.
  The theory that leverage should not affect firm value in perfect markets; with corporate
  taxes and bankruptcy costs, leverage increases the cost of equity, so in a rational pricing
  model *more leverage → higher expected equity returns* — the **opposite** of what PRT find.

## The anomaly in the broader literature

- **George, T. & Hwang, C. (2010).** *A Resolution of the Distress Risk and Leverage
  Puzzles in the Cross Section of Stock Returns.* Journal of Financial Economics, 96(1), 56–79.
  A decomposition of why leverage is negatively related to returns: they argue it reflects
  financial distress costs, not mispricing.

- **Novy-Marx, R. (2013).** *The Other Side of Value: The Gross Profitability Premium.*
  Journal of Financial Economics, 108(1), 1–28.
  High profitability firms tend to be lower-leveraged and outperform; the leverage anomaly
  is partially a profitability proxy.

- **Fama, E. & French, K. (1992).** *The Cross-Section of Expected Stock Returns.*
  Journal of Finance, 47(2), 427–465.
  The foundational cross-sectional paper; leverage is actually *positively* related to
  returns in early samples (supporting MM risk compensation) — a counterpoint to PRT.

- **Fama, E. & French, K. (2015).** *A Five-Factor Asset Pricing Model.*
  Journal of Financial Economics, 116(1), 1–22.
  The investment factor (CMA) is related to leverage changes; profitable firms invest
  less aggressively and tend to have lower leverage — linking leverage to the investment
  anomaly in a risk-factor framework.

## Survivorship bias in EDGAR-based studies

- **Survivorship bias in this study.**  The desk's shared EDGAR caches cover only the
  *current* S&P 500 membership projected backwards.  Firms that were acquired, went
  bankrupt, or were otherwise delisted are excluded.  High-leverage firms are
  over-represented among delistings; excluding them biases against finding the PRT anomaly
  (the highest-risk high-leverage outcomes are missing from the high-leverage bucket).
  The fact that we still do not find a significant signal even with this bias in our
  favour makes the null more credible.  The canonical treatment of survivorship bias in
  published research: Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias in
  Performance Studies*, Review of Financial Studies, 5(4), 553–580.

## Method lineage (the desk's shared engine)

- **Newey-West HAC t-stat.**  Newey, W.K. & West, K.D. (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,*
  Econometrica, 55(3), 703–708.  Used in `strategy.summarize` for the short annual spread
  series.
- **Annual quintile sorts.**  Following the standard cross-sectional factor literature
  (Fama-French, Novy-Marx, etc.): sort at fiscal year-end, hold for one calendar year,
  equal-weight portfolios.
- **Random-portfolio null.**  The null that any concentrated quintile-size random subset
  might outperform by luck; see also Fama & French (1993), *Common Risk Factors in the
  Returns on Stocks and Bonds*, Journal of Financial Economics.

## Related desk studies

- **[Study 65 — Scorecard](../../65-scorecard/)**: the low-risk / low-beta anomaly
  (Frazzini & Pedersen 2014) — a related phenomenon where "safer" firms outperform.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's combined
  quality + cheapness rank, which uses leverage implicitly in the enterprise value
  denominator.
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)** and
  **[Study 53 — Jackpot](../../53-jackpot/)**: other EDGAR-based factor studies with
  the same survivorship-bias caveat and similar methodology.
