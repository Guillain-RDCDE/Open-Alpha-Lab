# References & literature map — Study 664 (ESG Premium)

## The claim under test

- **The folklore.** "Doing well by doing good" — ESG-screened / ESG-tilted portfolios either
  earn a **premium** (better-governed, less-risky companies outperform) or, in the harsher
  telling, cost you performance (you're paying an insurance premium against tail/regulatory
  risk, and diversification is worse because the investable universe is smaller). Both framings
  are common in retail marketing and financial-media commentary.
- **The academic record is genuinely mixed.** Friede, Busch & Bassen (2015, *ESG and financial
  performance: aggregated evidence from more than 2000 empirical studies*, Journal of
  Sustainable Finance & Investment) find a slight positive tilt in the meta-literature, but
  warn about publication bias and heterogeneous methodology. Bolton & Kacperczyk (2021,
  *Do investors care about carbon risk?*, JFE) find a *carbon premium* — high emitters earn
  **higher** returns, i.e., ESG exclusion of high-carbon names should be a modest **drag**, not
  a boost, on pure risk-premium grounds. Pástor, Stambaugh & Taylor (2021, *Sustainable
  investing in equilibrium*, JFE) build a theory where ESG *demand* can produce positive
  realized alpha in a transition period even though expected alpha is negative in equilibrium —
  i.e. the sign can flip depending on the sample window. In short: there is no settled academic
  consensus that a durable ESG premium exists; the credible range runs from "small drag" to
  "small, transitional boost."
- **The mechanical counter-story.** ESG screens (excluding fossil fuels, tobacco, weapons,
  gambling) mechanically underweight Energy, Utilities and Materials and overweight
  Technology and Communication Services — the same sectors that dominate any large-cap
  growth/quality tilt over the last decade. Kacperczyk et al. and multiple practitioner notes
  (MSCI, iShares fund literature) document this "ESG = growth beta" mechanism directly.

## What we measure, and the honesty rails

- **Tracking difference & Sharpe** — CAGR, annualized vol and **excess-of-cash Sharpe** (both
  legs measured against the same ^IRX 13-week T-bill proxy — never a raw-vs-excess race) for
  ESGU vs SPY (2016-12 inception → as-of) and SUSA vs IVV (2005-01 inception → as-of).
  Tracking error and information ratio quantify the basis risk carried for whatever the active
  return turns out to be.
- **Active-return spread, Newey-West primary.** Daily (fund − benchmark) return is
  autocorrelated (both legs are highly correlated large-cap total-return series), so the
  **planned primary** is a Newey-West (1987) 5-lag HAC *t* of the mean; a Welch *t* is reported
  as a cross-check. Costs are one documented convention: 2 legs × one-way cost (5 bps) at entry
  and exit, amortized over the sample, plus a 30 bps/yr short-borrow drag on the benchmark leg
  (a standard, liquid large-cap-ETF borrow rate) — never charged twice, never silently dropped.
- **Factor decomposition, the decisive test.** Fund return regressed on [benchmark return,
  growth−value spread (IVW−IVE), quality spread (QUAL−benchmark)] with Newey-West HAC SEs. The
  regression intercept is the "ESG alpha" once both the market and the two dominant large-cap
  style tilts are priced in — this is how we test the "it's just a growth/quality tilt"
  counter-claim directly, not by assertion.
- **No survivorship on the Signal axis.** Every ticker used (ESGU, SUSA, SPY, IVV, IVW, IVE,
  QUAL) is a single, currently-traded ETF over its own full listed history — there is no
  cross-sectional panel of individual constituents and therefore no survivor-bias basket to
  name. (Contrast with sibling study 211-sin-stocks, which *does* build a small hand-picked
  stock basket and names that bias explicitly.)
- **Execution convention (single, documented lag).** None needed for the tracking/spread tests
  — both legs use the same daily close-to-close bar, so there is no signal-to-execution lag to
  document (this is a scheduled, always-on comparison, not a signal-triggered entry).

## Data sources

- **ESGU, SUSA, SPY, IVV, IVW, IVE, QUAL daily adjusted (total-return) closes** and **^IRX
  (13-week T-bill discount yield)** — yfinance (no key), cached under `_cache/`
  (`esgprem_<ticker>.csv`), 2004-01-02 → 2026-06-30 (each series starts at its own inception).
- **Fund facts (inception dates, expense ratios)** — hardcoded in
  [`esg_premium/data.py`](../esg_premium/data.py) from issuer fact sheets (ishares.com,
  ssga.com); stable, well-documented figures, no fitting.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [211-sin-stocks](../../211-sin-stocks/) — the **anti-ESG** side: a hand-picked "sin" basket
  (tobacco, alcohol, gambling, defense) vs the market and vs DSI (an ESG counter-portfolio).
  That study tests the **Hong-Kacperczyk neglect premium** (do *shunned* stocks outperform
  because they're under-owned?) using individual stocks and finds no certifiable premium
  either (excess-return HAC *t* = −0.46). This study tests the **mirror claim** — do the
  ESG-*included* funds (ESGU, SUSA) outperform — using the actual flagship ESG fund products,
  not a hand-picked stock basket, and adds the factor decomposition 211 doesn't attempt.
- [200-roe-quality](../../200-roe-quality/) — the **quality factor itself** (profitability /
  ROE sorts) as a standalone anomaly. This study *uses* a quality-factor ETF (QUAL) purely as
  a **control variable** to explain away any apparent ESG edge — it does not re-test whether
  quality investing works on its own terms; see 200 for that.
- [246-defensive-sectors](../../246-defensive-sectors/) — low-beta/defensive **sector**
  rotation (utilities, staples, healthcare). ESG screens move in the *opposite* sectoral
  direction from that study (ESG funds tend to underweight utilities, a defensive sector, for
  environmental-screen reasons) — a different mechanism, no ticker overlap.
- [335-buzz-sentiment-etf](../../335-buzz-sentiment-etf/) and
  [334-ark-innovation](../../334-ark-innovation/) — **thematic / sentiment-driven** ETF
  products (social-buzz baskets, disruptive-innovation growth stocks). Both are explicitly
  *active*, concentrated, high-turnover growth vehicles; ESGU/SUSA are broad, low-tracking-
  error, near-full-market-cap ESG-*screened* index funds — a fundamentally different
  construction, tested here for a fundamentally different claim (a passive labeling premium,
  not active stock-picking skill).

None of the siblings test the two flagship large-cap ESG ETF products against their own
plain-vanilla benchmarks with a growth/quality factor decomposition — that is this study's own
axis.
