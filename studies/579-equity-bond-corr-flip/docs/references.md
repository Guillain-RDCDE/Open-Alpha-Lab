# References & literature map — Study 579 (Equity-Bond-Corr-Flip)

## The claim, at full strength

- **Campbell, Sunderam & Viceira (2017)**, *"Inflation Bets or Deflation Hedges? The Changing
  Risks of Nominal Bonds."* *Critical Finance Review* 6(2). The canonical model of why the
  stock-bond correlation *switches sign*: when macro shocks are dominated by **growth/deflation**
  risk, bonds hedge equities (negative correlation) and command a low or negative risk premium;
  when **inflation** risk dominates, bonds and stocks fall together (positive correlation). The
  mechanism behind the 2022 flip.
- **Baele, Bekaert & Inghelbrecht (2010)**, *"The Determinants of Stock and Bond Return
  Comovement."* *Review of Financial Studies* 23(6). Documents the time-varying stock-bond
  correlation and shows liquidity/flight-to-safety and macro factors — not a constant — drive it.
- **Ilmanen (2003)**, *"Stock-Bond Correlations."* *Journal of Fixed Income* 13(2). The
  practitioner statement that the sign and level of the stock-bond correlation is *the* input to
  whether bonds diversify equities — and that it is regime-dependent, not structural.
- **Brixton, Brooks, Hecht, Ilmanen, Maloney & McQuinn (2023, AQR)**, *"A Changing Stock-Bond
  Correlation: Drivers and Implications."* *Journal of Portfolio Management*. The post-2022
  reassessment: the correlation regime matters for risk-parity / 60/40 sizing, but timing it
  reliably is hard — the practitioner ambivalence this study quantifies.

## The 2022 event and the 60/40

- The 2022 drawdown of a US 60/40 book (roughly −16% to −18% total return, the worst since the
  1930s for a balanced portfolio) is the single macro observation motivating "the correlation
  sign tells you when 60/40 stops working." This study measures how much of a *tradable* signal
  survives once the whole 24-year tape — not just 2022 — is on the same clock.

## Neighbours on this bench (the dedup map)

- **[Study 578 — Cross-Asset-Correlation-Regime](../../578-cross-asset-correlation-regime/)** — the
  *broad* cross-asset correlation regime (average pairwise correlation across many assets) as a
  risk-on/risk-off state. Study 579 is the **specific** SPY/TLT sign and its use as a **60/40
  hedge timer** — one pair, one portfolio, the diversification question.
- **[Study 502 — Betting-Against-Correlation](../../502-betting-against-correlation/)** — a
  *cross-sectional* stock-level signal (low-correlation names). Study 579 is a *time-series* macro
  regime on a single equity/bond pair.
- **[Study 245 — Oil-Equity-Correlation](../../245-oil-equity-correlation/)** — the oil↔equity
  contemporaneous-vs-predictive split. Same correlation-as-signal family, different assets and a
  lead/lag (not a regime-sign) question.
- **[Study 152 — Inflation-Hedge](../../152-inflation-hedge/)** /
  **[Study 119 — Real-Rate-Regime](../../119-real-rate-regime/)** — the macro drivers (inflation,
  real rates) *behind* the correlation flip; this study tests the flip's tradability directly.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the negative-minus-positive
  regime spread on forward returns.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  correlation-sign labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on the *real* tape for `REAL`; literature/mechanism support alone reads `WEAK`), one
  documented execution lag, gross-and-net labeling, and the low effective-N caveat named on the
  Signal axis (one regime change ≠ many observations).
