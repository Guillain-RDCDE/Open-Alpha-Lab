# References & literature map — Study 220 (Small Dogs of the Dow)

## The claims under test

The **Small Dogs of the Dow** (also called the "Low-5"): of the ten highest-yielding Dow
stocks, take only the five with the *lowest absolute share price* each January — equal-weight,
hold a year, repeat. The **Foolish Four** is a further variant popularised by the Motley Fool
in the mid-1990s: of the same ten Dogs, skip the #1 cheapest name and buy the next four
equal-weight. Both sub-strategies claim to beat not just the Dow but even the full Dogs basket.

## Primary sources on the original Dogs strategy

- Michael O'Higgins & John Downes, *Beating the Dow* (HarperCollins, 1991) — the book that
  introduced the Dogs of the Dow; the Small Dogs / Low-5 sub-screen appears in later editions.
- Michael O'Higgins with John Downes, *Beating the Dow with Bonds* (HarperCollins, 1999) —
  updates the Dogs and Low-5 performance record through the late 1990s (the peak of the
  in-sample window the price filter was optimised on).

## The Foolish Four — rise and fall

- The Motley Fool website (David & Tom Gardner, ~1996) popularised the "Foolish Four" as the
  best of the Dogs variants. The rule (skip the cheapest, hold #2-#5 by price) was discovered
  by sorting historical returns rather than by theory — a textbook example of data-snooping.
- Hubert & Margevicius, "The Foolish Four: Fact or Fiction?" — Motley Fool internal analysis
  (1999-era) acknowledging the out-of-sample collapse.
- The Motley Fool officially retired the Foolish Four strategy in January 2000, publishing a
  candid post-mortem: the extra in-sample edge from the price filter disappeared almost
  immediately in live trading (from their published archive, circa 2000-01).

## Statistical challenges to the Dogs / Small Dogs edge

- Grant McQueen, Kay Shields & Steven Thorley, *"Does the 'Dow-10 Investment Strategy' Beat
  the Dow Statistically and Economically?"*, Financial Analysts Journal 53(4), 1997. —
  Rigorous early test showing the Dogs-10 gap is not statistically significant once you control
  for risk, size and dividend-tax drag. Direct precursor to this study's design.
- Dale Domian & David Louton, *"Is Bigger Better? Tests of the Small Dogs Investment
  Strategy"*, Journal of Financial Research 21(3), 1998. — Directly tests the Low-5 / Small
  Dogs; finds the price-filter extra return does not survive risk adjustment or out-of-sample
  testing; interprets the residual gap as a small-cap / low-price premium, not Dogs alpha.
- Visscher & Filbeck, *"Dividend-Yield Strategies in the Canadian Stock Market"*, Financial
  Analysts Journal 59(1), 2003. — Applies the Dogs framework to an alternative market; finds
  the yield-based selection does not generalise robustly.

## Why the price filter might have an economic interpretation

- Fama & French, *"The Cross-Section of Expected Stock Returns"*, Journal of Finance 47(2),
  1992. — Value/size premium: low-priced Dow names are implicitly a small-cap / distressed
  tilt within the large-cap universe; the Dogs price filter is a crude proxy for the size
  factor, not an independent signal.
- DeBondt & Thaler, *"Does the Stock Market Overreact?"*, Journal of Finance 40(3), 1985. —
  Mean-reversion in "loser" stocks: low-priced, beaten-down blue chips (the low-$-price tail
  of the Dogs) may benefit from a reversal premium — but the reversal literature requires
  multi-year holding, not a 12-month screen.

## Data-mining / multiple-comparison framing

- Lo & MacKinlay, *"Data-Snooping Biases in Tests of Financial Asset Pricing Models"*, Review
  of Financial Studies 3(3), 1990. — The general framework for why strategy variants
  discovered by sorting historical outcomes (i.e. the Foolish Four) should be penalised by
  multiple-comparison corrections.
- Harvey, Liu & Zhu, *"… and the Cross-Section of Expected Returns"*, Review of Financial
  Studies 29(1), 2016. — Documents that most published anomalies fail out-of-sample; the
  |t| ≥ 3 recommendation for truly new signals (this study uses the house |t| ≥ 2 bar for
  well-known, pre-registered hypotheses; the data-mining concern pushes toward the higher bar
  here).

## Inference methods

- Newey & West (1987), *"A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix"*, Econometrica — HAC standard errors for the
  annual excess series.
- Politis & Romano (1992/1994) — circular block bootstrap for short annual panels.

## Data sources

- **Dow components**, daily total-return adjusted (`yfinance`, `auto_adjust=True`) + per-share
  dividend stream, via the Study 88 point-in-time membership timeline. Reuses Study 88's
  cached parquet files (`88-dogs-of-the-dow/_cache/`).
- **Benchmark: DIA** (SPDR Dow Jones Industrial Average ETF), total return — same-basis
  yardstick; DIA inception 1998.

## Related desk studies

- [Study 88 — Dogs-of-the-Dow](../../../88-dogs-of-the-dow/) — the parent study; this study
  inherits its point-in-time membership data and verdict (Weak/Fragile) as the baseline.
- [Study 196 — Long-Term-Reversal](../../../196-long-term-reversal/) — DeBondt-Thaler
  reversal as a cleaner version of the "buy beaten-down names" thesis.
- [Study 46 — Bargain-Bin](../../../46-bargain-bin/) — low absolute price within a different
  universe; similar concerns about the price filter as a crude size proxy.
