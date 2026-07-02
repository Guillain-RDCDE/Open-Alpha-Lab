# References & literature map — Study 548 (Happiness-Index-Country)

## The claim & its ingredients

- **Helliwell, Layard, Sachs, De Neve et al. (2024)**, *World Happiness Report 2024.* The source of
  the country happiness **rank** (1 = happiest) this study sorts on. WHR ranks countries by a
  Cantril-ladder life-evaluation survey, decomposed into six factors (GDP per capita, social
  support, healthy life expectancy, freedom, generosity, perceptions of corruption). The methodology
  has shifted across editions — one reason a clean point-in-time happiness × returns panel does not
  exist.
- **The folklore.** "Optimistic, well-governed, high-trust societies must have better stock markets"
  is a recurring alt-data / financial-media story (a cousin of the sentiment-and-returns genre). It
  has no canonical academic backing at the *tradable single-country-index* level — which is precisely
  why it is a good spurious-correlation demonstration.

## Why this is a spurious-correlation demo

- **Vul, Harris, Winkielman & Pashler (2009)**, *"Puzzlingly High Correlations in fMRI Studies of
  Emotion, Personality, and Social Cognition."* The canonical warning that eye-catching
  cross-sectional correlations on small samples are routinely artefacts — the statistical spirit of
  this study.
- **Vigen (2015)**, *Spurious Correlations.* The pop-culture catalogue of nonsense cross-series
  correlations; a happiness-rank × country-return sort on n = 24 is exactly this genre.
- **Harvey, Liu & Zhu (2016)**, *"…and the Cross-Section of Expected Returns."* *Review of Financial
  Studies* 29(1). Multiple-testing inflation in the factor zoo: with enough candidate cross-country
  sorts, some will "work" by chance. A single lucky window is not a signal.

## The measure & method we build

- **Rank-sorted long-short spread.** Sort the investable cross-section by WHR rank, long the happiest
  tercile, short the gloomiest, and read the forward-return spread.
- **Spearman rank correlation** (Spearman 1904) — the single number the folklore lives on: rank
  happiness against forward return; a real effect shows a strong, stable, significant rho.
- **Welch (1947)** — the unequal-variance two-sample *t* for the happy-minus-gloomy bucket spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  happiness labels against forward returns and read the spread's tail probability.

## Neighbours on this bench (the dedup map)

- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)**, **[Study 335 —
  Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)**, **[Study 300 —
  Sports-Sentiment](../../300-sports-sentiment/)** — *sentiment/mood* signals that time or tilt a
  market. Study 548 differs in axis: it is a **cross-country** sort on a *societal* well-being index,
  not a time-series mood indicator on one tape.
- **[Study 273 — Lego-Returns](../../273-lego-returns/)**, **[Study 276 —
  Sneaker-Resale](../../276-sneaker-resale/)** — the alt-data / synthetic-only cousins where a free,
  clean, tradable real panel does not exist, so the study is capped below REAL and the
  data-availability limitation is named on the Signal axis. Study 548 shares that structure (tiny
  investable cross-section, no aligned panel).

## Shared method

- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a real tape for REAL, else WEAK/NONE), the placebo null and seed-robust synthetic
  positive control (≥ 20 seeds), one execution lag, costs one-way × NAV with shorts paying borrow,
  and the small-N / no-real-panel limitation named on the Signal axis.
