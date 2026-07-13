# References & literature map — Study 748 (CEO-Age-Effect)

## The claim, at its strongest (this one has real papers)

- **Serfling, M. A. (2014), "CEO age and the riskiness of corporate policies," *Journal of
  Corporate Finance* 25, 251–273.** The anchor. Older CEOs are found to run *less* risky
  policies — lower R&D intensity, more diversifying acquisitions, less leverage, smoother
  operations. This is the "old CEO plays it safe" half of the folklore, empirically grounded.
- **Yim, S. (2013), "The acquisitiveness of youth: CEO age and acquisition behavior,"
  *Journal of Financial Economics* 108(1), 250–273.** The "young CEO is aggressive" half:
  younger CEOs make significantly more acquisitions, and the market reacts to age. The pop
  translation ("young bosses swing for the fences") comes from here.
- **Where the *trade* over-reaches.** Both papers measure *corporate policies and behaviour*,
  not a tradable cross-sectional *stock-return* premium. The leap — "so buy young-CEO firms,
  short old-CEO firms" — is the folklore this study tests and busts. Riskier corporate policy
  need not (and here does not) produce risk-adjusted return; the low-risk anomaly literature
  (below) suggests the opposite direction if anything.

## Why the trade fails — the confound and the factor it really loads

- **Omitted-variable / factor confound.** "Young CEO" is nearly collinear with "founder-led
  growth-tech that IPO'd recently." The long-young/short-old book is therefore predominantly a
  **growth-vs-value / high-beta** bet. Fama, E. F. & French, K. R. (1993), "Common risk factors
  in the returns on stocks and bonds," *JFE* 33 — the market/size/value factors the spread
  actually loads. The honest test is the CAPM **alpha** once the market is regressed out.
- **The low-volatility / betting-against-beta anomaly.** Frazzini, A. & Pedersen, L. H. (2014),
  "Betting against beta," *JFE* 111 — high-beta baskets (like the young-CEO cohort here) tend to
  *underperform* risk-adjusted, which is exactly why the young basket's Sharpe (0.90) trails the
  old basket's (1.08) despite a higher raw return.
- **Sub-period / regime instability = factor exposure.** A "signal" whose sign flips with the
  macro regime (young-CEO growth: +27%/yr in 2018–20, −34%/yr in the 2021–22 rate shock, +17%/yr
  in the 2023–26 AI melt-up) is the fingerprint of a factor tilt, not a firm-characteristic alpha.

## Shared method (the desk's engine)

- **Newey, W. K. & West, K. D. (1987, 1994).** The heteroskedasticity- and autocorrelation-
  consistent (HAC) covariance and the automatic Bartlett-kernel lag rule — the desk's
  autocorrelation-robust *t* on the mean monthly long/short and on the CAPM alpha
  (`strategy.hac_mean_t`, `strategy.capm_alpha`). `REAL` requires |t| ≥ 2 here.
- **Label-shuffle / permutation testing** (Fisher 1935; Good, P., *Permutation, Parametric and
  Bootstrap Tests of Hypotheses*, 2005) — the placebo null: reshuffle the young/old labels across
  names and read the HAC |t|'s tail probability (`strategy.placebo_pvalue`).
- **Sharpe, W. F. (1966, 1994)** — the risk-adjusted yardstick that reverses the raw-return
  ranking (old CEOs win per unit of risk).
- **Seed-robust synthetic positive control (≥ 20 seeds).** `data.synthetic_panel` plants a young
  higher-beta AND a genuine `age_alpha`; the engine must certify a *planted* premium and stay flat
  at the null (`strategy.synthetic_mean_alpha_t`) — control only, never cited for the real stamp.

## Neighbours on this bench (the dedup map)

- **[Study 543 — Western-Zodiac-CEO](../../543-western-zodiac-ceo/)** — the sibling *CEO
  characteristic* sort (sun sign). There the characteristic is astrological noise; here it is a
  real trait (age) whose apparent edge turns out to be a factor confound. Both cap below `REAL` on
  a curated ~40-name table.
- **[Study 391 — CEO-Turnover](../../391-ceo-turnover/)** — the *event* cousin (firing the CEO):
  a short-window abnormal-return study on a hardcoded, labelled table. Study 748 is the
  *cross-sectional characteristic* version (who the CEO *is*, not what happens to them).

## Data sources used here

- **Hardcoded CEO → birth-year table** (`ceo_age_effect.data.CEO_AGES`): ~40 large-cap CEOs with a
  clean single-person tenure over the sample, birth years from public sources (Wikipedia / company
  bios / SEC filings / press). ExecuComp/BoardEx age panels are not free; the transparent, cited
  table is the stand-in. The table is not survivorship-free and the young bucket skews to recent
  IPOs — a **listing-vintage / survivorship tilt named on the Signal axis**.
- **yfinance** daily **total-return** (dividend-adjusted) closes for each ticker + SPY, resampled
  to monthly, cached under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (`REAL` needs a
  robust *t* ≥ 2 on the real tape; literature/curated tables cap at `WEAK`/`NONE`), one documented
  execution lag, costs one-way × NAV with shorts paying borrow, gross/net and total-return labelled.
