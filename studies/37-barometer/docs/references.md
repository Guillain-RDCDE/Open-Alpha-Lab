# References & literature map — Study 37 (Barometer)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entries are **§19.2 (fundamental /
  macro-driven strategies)** — trading assets on the trend in macroeconomic fundamentals (growth,
  inflation, monetary policy) — and **§19.3 (inflation hedging)** — tilting a portfolio toward real
  assets that protect purchasing power when inflation rises. *(Copyrighted; not redistributed.)*

## The claim under test — the steelman

- **Brooks, J. & Moskowitz, T. (2017), "Macro Momentum: Returns Predictability Across Asset Classes."**
  (AQR / SSRN [2949379](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2949379).) The foundational
  statement of the idea: the *trend* in fundamental macro data — growth, inflation, monetary policy,
  risk sentiment — predicts returns across equity indices, bonds, currencies and commodities, and a
  diversified macro-momentum strategy delivers a positive, low-correlation premium. Real, but modest and
  slow, with long flat stretches — the basis for the `WEAK`/`REAL` signal stamp.
- **Neville, H., Draaisma, T., Funnell, B., Harvey, C. R. & van Hemert, O. (2021), "The Best Strategies
  for Inflationary Times," *Journal of Portfolio Management* 47(8)** (SSRN
  [3813202](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3813202)). The steelman for the
  inflation hedge: across the inflationary regimes of the last century, nominal stocks and bonds do
  *badly*, while **commodities, real assets, and trend-following** earn strongly positive real returns —
  so tilting toward real assets when inflation is rising protects a portfolio in exactly the regimes that
  hurt it most. Crucially it is **episodic**: inflationary regimes are rare, so the hedge is a drag most
  of the time — the basis for the `FRAGILE` tradability stamp and the beat-7 regime split.

## Why the macro premium exists — the economics

- **Moskowitz, T., Ooi, Y. H. & Pedersen, L. (2012), "Time Series Momentum," *Journal of Financial
  Economics* 104(2).** Trend persistence across asset classes — the price-based cousin of macro
  momentum; both are slow, diversifying, crisis-friendly premia (see Study 31 Trade-Winds).
- **Ilmanen, A. (2011), *Expected Returns* (Wiley).** A book-length map of how growth, inflation and
  liquidity regimes price every major asset — the backdrop for trading the *change* in those regimes.

## The regime machinery — and why the inflation hedge must be conditional

- **Ang, A. & Bekaert, G. (2002), "International Asset Allocation with Regime Shifts," *Review of
  Financial Studies* 15(4)** (and Ang & Bekaert 2004, "How Regimes Affect Asset Allocation," *FAJ*).
  Returns and correlations are regime-dependent, so a hedge designed for one regime (rising inflation)
  must be *evaluated conditionally* — the formal basis for the beat-7 split that asks whether the
  inflation tilt pays specifically when inflation is rising.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference** (Newey & West, *Econometrica* 1987) on the books' means.
- **Data.** The real run *would* use monthly **FRED macro series** (growth: `INDPRO` / `PAYEMS`;
  inflation: `CPIAUCSL` / `T10YIE`) plus liquid asset proxies; pinned with
  [`quantlab.repro`](../../../quantlab/repro.py). The synthetic control bakes two persistent,
  regime-switching latent macro states whose lagged momentum drives a small cross-asset panel.

## Caveats stated in the open (house rule)

- **Real run is PENDING a reliable FRED macro fetch.** In this environment the daily FRED series time out
  and even monthly CPI is intermittent, so the real cross-asset run is not yet populated — mirroring
  [Study 27 (Steamroller)](../../27-steamroller/)'s pending-fetch pattern. The committed verdict rests on
  the fully-validated synthetic control and the literature until a reliable fetch lands. This is a
  **pre-registration**: the apparatus, the null and the regime split are fixed before the real numbers.
- **Monthly horizon, simplified macro state.** The synthetic uses two latent drivers (growth, inflation);
  the real-world macro-momentum literature uses several (incl. monetary policy and risk sentiment) — a
  stated simplification that keeps the control legible while preserving the mechanism.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
