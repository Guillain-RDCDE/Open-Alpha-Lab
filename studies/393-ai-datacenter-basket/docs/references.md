# References & literature map — Study 393 (AI-Datacenter-Basket)

## The claim under test

- **The pitch (the "picks-and-shovels" framing).** A wave of 2024-2026 financial media,
  brokerage research and retail/social commentary reframed the AI trade away from "pick the
  one AI winner" and toward **"buy the build-out"**: the chips (NVDA, AVGO, AMD), the electrical
  and cooling gear that fills a datacenter (Vertiv `VRT`, Eaton `ETN`), the merchant-power
  utilities that sell the electricity an AI datacenter consumes (Constellation `CEG`, Vistra
  `VST`), the AI server vendors (Super Micro `SMCI`, Dell `DELL`), and the datacenter switch
  fabric (Arista `ANET`). The slogan is *"you don't have to know who wins — just own the
  datacenter-and-power basket and ride the capex."* The construction echoes the classic
  "sell shovels in a gold rush" heuristic (popularised around the dot-com and again the
  crypto-mining build-outs).
- **Why it's seductive.** Hyperscaler capex guidance (Microsoft, Alphabet, Amazon, Meta) and
  the resulting power-demand narrative are genuinely large and genuinely documented; the *theme*
  is real. The leap the pitch makes is from "AI capex is huge" to "**this specific eight-name
  basket** is the way to harvest it" — a leap that is only obvious *after* these eight have
  already re-rated.

## The trap this study isolates — selection after the fact

- **Naming the basket requires the outcome.** Exactly these eight tickers become "the obvious
  datacenter-and-power basket" only once you already know NVDA 30×'d, VRT and the merchant-power
  utilities re-rated, and SMCI/DELL caught the server cycle. In 2019 (or even early 2022) the
  same investor staring at the same field — semis, networking, electrical/industrials, regulated
  *and* merchant utilities, datacenter REITs — had no way to single out these eight in advance.
  This is the **look-ahead / survivorship** selection at the heart of every thematic basket.
- **The right null is a random basket from the same field.** Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns* (Review of Financial Studies), and Bailey &
  López de Prado (2014), *The Deflated Sharpe Ratio*, formalise why a strategy *selected* on its
  realised performance needs a far higher bar than a naive *t*-stat: the selection itself
  manufactures apparent edge. The operational version of that correction here is the **ex-post
  "pick the k winners"** placebo and the **random-basket sampling distribution** — where does a
  *blindly* chosen eight from the same field land?
- **Theme beta vs name selection.** Even granting the theme, "datacenter-and-power stocks went
  up" is mostly a *sector/theme beta*, not basket alpha. We strip it with an **equal-weight field**
  control (1/N of the whole candidate universe), so the residual basket spread is name selection
  *within* the theme, not "the theme rose." Cf. the factor-zoo critique (Cochrane, 2011,
  *Presidential Address: Discount Rates*).

## Why these benchmarks

- **SPY (the market) and QQQ (the tech tape).** The honest comparison for an AI-themed basket is
  not just the broad market but also a *concentrated-tech* benchmark — much of the basket's
  return is large-cap tech beta you could have owned in QQQ. We race the basket against both,
  excess-vs-excess (both legs fully invested, so the raw monthly difference *is* the excess), with
  an autocorrelation-robust (Newey-West) *t* on the spread.

## Method lineage (the desk's shared engine)

- **HAC (Newey-West) t-stat of the spread.**
  [`strategy.hac_tstat_diff`](../ai_datacenter_basket/strategy.py) — the Signal-axis test: the
  per-month basket-minus-benchmark difference under autocorrelation-robust inference (Newey &
  West, 1987, *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*).
- **Ex-post selection placebo + random-basket distribution.**
  [`strategy.expost_winners`](../ai_datacenter_basket/strategy.py) and
  [`strategy.random_baskets`](../ai_datacenter_basket/strategy.py) — the look-ahead "pick the
  winners" rule made explicit, and the blind-pick sampling distribution it is judged against
  (Fisher randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Single-name decomposition.**
  [`strategy.single_name_tstats`](../ai_datacenter_basket/strategy.py) — is the "basket" really one
  or two names (NVDA, VRT) wearing a theme's clothes?
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../ai_datacenter_basket/data.py) plants a known per-name tilt
  (``alpha_spread``) on a *pre-named* basket; the offline core runs with no network. The control
  proves the harness finds a spread *only* where one is planted (pre-named) **and** manufactures
  one *whenever* you select on the outcome (ex-post) — even at zero true edge.

## Data sources used here

- **yfinance** daily auto-adjusted (≈ total-return) closes, resampled to monthly returns, for a
  31-name datacenter/power candidate field + SPY + QQQ, 2019-01 → 2026-05; the common window after
  trimming to names with usable history is **2022-02 → 2026-05** (post-spin tickers CEG/VST set the
  start). Cached under `_cache/datacenter_panel.parquet`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 355 — Magnificent-Seven](../355-magnificent-seven/)**: the same selection-artefact
  pattern on the mega-cap "Mag 7." Hold the names that already won and you reproduce the
  random-basket distribution forward, not the in-sample spread. This study is its thematic cousin.
- **[Study 356 — GLP-1 basket](../356-glp1-basket/)**: another "buy the theme basket" trade
  (Ozempic/weight-loss) whose alpha collapses to one name and post-hoc selection.
- **[Study 350 — Dartboard-Portfolio](../350-dartboard-portfolio/)** and
  **[Study 345 — Survivorship-Bias](../345-survivorship-bias/)**: the mechanics of selection and
  survivorship that this basket inherits by construction.
