# References & literature map — Study 356 (the GLP-1 / Ozempic basket)

## The claim under test

- **The viral theme.** Across financial media and retail forums (2023–2024) the *"Ozempic
  trade"* / *"GLP-1 trade"* became a named basket: buy **Eli Lilly (LLY)** and **Novo Nordisk
  (NVO)** — the makers of tirzepatide (Mounjaro/Zepbound) and semaglutide (Ozempic/Wegovy) —
  to ride the structural boom in weight-loss drugs, sometimes broadened to a "GLP-1 winners /
  losers" basket (food, dialysis, medical-device shorts). The testable claim: the basket
  delivers **alpha** over the market and over healthcare, i.e. it is more than the sector and
  more than two lucky single names.
- **Where it's made.** Sell-side thematic notes (e.g. Morgan Stanley's obesity-drug TAM
  upgrades, 2023), the launch of dedicated obesity/GLP-1 thematic ETFs (2023–2024), and
  countless "how to play Ozempic" retail explainers. The theme is real as a *narrative*; the
  question is whether it's an *edge*.

## Why "it beat the market" is not yet alpha

- **The excess-return / market-model identity.** A basket's edge is its return *net of the
  benchmark's* (mean excess) and, more strictly, the intercept α in `r = α + β·mkt + ε`
  (Sharpe 1964; Lintner 1965; Jensen 1968, *The Performance of Mutual Funds 1945–1964*, which
  introduced Jensen's α). A high raw return with high β is mostly *paid-for* market exposure,
  not skill. We report both the mean excess and the market-model α vs SPY **and** vs XLV.
- **Autocorrelation-robust inference.** Daily returns are heteroskedastic and serially
  correlated, so the i.i.d. *t* overstates significance. Newey & West (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, gives the HAC standard errors we use for every *t* on the Signal axis; the
  truncation lag follows the automatic rule `floor(4(n/100)^(2/9))`.
- **Block bootstrap for the CI.** Politis & Romano (1994), *The Stationary Bootstrap*; Künsch
  (1989), *The Jackknife and the Bootstrap for General Stationary Observations*. We resample
  21-day blocks so the excess-return CI respects the volatility clustering an i.i.d. bootstrap
  would destroy.

## Why this is concentration / recency, not a factor

- **Post-hoc selection & the winners' bias.** LLY and NVO are in the basket *because* they
  won the GLP-1 race — a textbook look-ahead/recency selection. Brown, Goetzmann, Ibbotson &
  Ross (1992), *Survivorship Bias in Performance Studies*, show how selecting on realised
  success manufactures apparent alpha. A "theme basket" named after the product whose makers
  already 10×'d is selection by construction; the bias points **up**, inflating the excess.
- **Concentration ≠ diversified factor.** Two names with pairwise correlation ~0.40 is not a
  diversified premium; it is idiosyncratic, path-dependent single-stock risk (here a −49%
  basket drawdown, −75% in NVO alone). Standard diversification math (Markowitz 1952,
  *Portfolio Selection*) says a 2-name book carries large idiosyncratic variance that no
  amount of "theme" removes.
- **Theme/narrative investing and its weak record.** Barberis & Shleifer (2003), *Style
  Investing*, and the post-2021 thematic-ETF experience (many launched at the top of their
  story and underperformed) frame the prior: by the time a theme is a named, ETF-ified trade,
  the cheap part is gone. Our recency split — positive excess *before* 2023, negative *during*
  the 2023–2026 mania — is exactly that pattern.

## Data sources used here

- **Yahoo Finance** (`yfinance`), daily **adjusted close** (total-return) for LLY, **NVO (US
  ADR)**, XLV (Health Care Select Sector SPDR), SPY, 2018-01-02 → 2026-06-18. Adjusted close
  reinvests dividends — labelled total-return, not price-only, on every table. NVO is the US
  ADR (the line a US reader actually trades); the Copenhagen B-share would differ by FX and
  ADR effects, named as a proxy choice. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Method lineage (the desk's shared engine)

- **HAC / market-model inference.** [`strategy.hac_t_mean`](../glp1_basket/strategy.py),
  [`strategy.market_model`](../glp1_basket/strategy.py) — Newey-West *t* on mean excess and on
  the CAPM α; the Signal axis requires HAC-t ≥ 2 on the *real* tape (literature support alone
  reads WEAK).
- **Deterministic synthetic control.** A fixed-seed one-factor panel
  ([`data.synthetic_panel`](../glp1_basket/data.py)) plants a known per-name α; the engine
  must recover it and the *absence* of it (NVO) — a machinery proof that runs with no network.

## Related desk studies

- **Survivorship / selection family** — studies that show picking on realised winners
  manufactures alpha; the same mechanism inflates any "theme that already won" basket.
- **Concentration & single-name risk** — the same lesson as any "just buy the winner" pitch:
  a two-name book is a bet on one path, not a harvestable, scalable premium.
