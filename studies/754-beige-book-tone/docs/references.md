# References & literature map — Study 754 (Beige-Book-Tone)

## The claim under test

- **The Beige Book.** Board of Governors of the Federal Reserve System,
  *Summary of Commentary on Current Economic Conditions by Federal Reserve District* (the
  "Beige Book"). Published **eight times a year**, on a Wednesday roughly **two weeks before**
  each FOMC meeting; a qualitative, anecdote-driven digest of business contacts across the
  twelve Reserve districts. Full archive: <https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm>.
- **The market-timing folklore.** Fed-watchers read the Beige Book's **tone** — the balance of
  upbeat vs downbeat language — as a mood reading on the economy, and the trading lore follows:
  a **positive-tone** book precedes an equity **drift up** in the days after release. The idea
  recurs across sell-side "Beige Book takeaways," financial-media recaps, and Fed-watching
  newsletters. We test the strongest form: *does a positive-tone Beige Book lead SPY cleanly
  enough, and early enough, to trade?*
- **The sentiment tool.** Loughran, T. & McDonald, B. (2011), *When Is a Liability Not a
  Liability? Textual Analysis, Dictionaries, and 10-Ks* (Journal of Finance 66(1), 35–65) —
  the finance-specific positive/negative word lists ("LM dictionary") that make text-tone
  scoring on financial documents credible (generic Harvard-IV lists mis-score finance text).
  Net tone here = $(\#\text{pos} - \#\text{neg})/(\#\text{pos} + \#\text{neg})$.

## Does the Beige Book carry information? — the honest prior

- **Beige-Book economic content.** Armesto, M. T., Hernández-Murillo, R., Owyang, M. T. &
  Piger, J. (2009), *Measuring the Information Content of the Beige Book: A Mixed Data Sampling
  Approach* (Journal of Money, Credit and Banking 41(1), 35–55) — a quantified Beige-Book tone
  does track **current and near-term real activity**, but adds little beyond other timely
  indicators. Balke & Petersen (2002), *How Well Does the Beige Book Reflect Economic Activity?*
  (JMCB 34) reach a similar read. The literature supports *economic* content — **not** an
  equity-market lead, which is the trading leap we isolate.
- **Central-bank communication & text tone.** Lucca, D. & Trebbi, F. (2009), *Measuring Central
  Bank Communication* (NBER WP 15367); and the broader finding that markets react to FOMC
  *statements/minutes* far more than to the Beige Book — the Beige Book lands ~2 weeks **before**
  the meeting that actually reprices rates, so its information is dominated by the coming
  decision.

## Why "leads" is the crux — description vs prediction

- **The stock market as its own leading indicator.** Stock prices are a Conference Board LEI
  component and famously lead the real economy (Samuelson's quip that the market "predicted nine
  of the last five recessions"). A text that *describes* current conditions therefore need not
  **lead** the price — it may merely echo a turn equities already made. This is the confound the
  study isolates via an **ex-2020 regime test** and a corr(tone, *prior* return) check.
- **Predictive regressions and small-sample caution.** Welch, I. & Goyal, A. (2008), *A
  Comprehensive Look at the Empirical Performance of Equity Premium Prediction* (Review of
  Financial Studies 21(4)) — most "predictors" that look significant in-sample fail honestly
  tested; the bar for a tradable macro/text signal is high.
- **Post-announcement drift & event studies.** MacKinlay, A. C. (1997), *Event Studies in
  Economics and Finance* (Journal of Economic Literature 35) — the standard close-to-close
  event-window methodology used here, with a strict release-date entry (no look-ahead).

## Why the inference is small-sample / HAC / placebo-based

- **Welch two-sample t.** Welch, B. L. (1947), *The generalization of "Student's" problem when
  several different population variances are involved* (Biometrika 34) — unequal-variance test of
  the positive-tone forward mean against the unconditional mean.
- **Newey-West HAC standard errors.** Newey, W. K. & West, K. D. (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica 55) — the Bartlett-kernel HAC *t* on the continuous tone→drift slope (releases
  are ~monthly, so multi-day windows barely overlap and OLS/HAC agree).
- **Randomization / placebo null.** Because events are few and autocorrelated, we resample random
  same-size event sets and ask how often chance is as bullish as the positive set (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).

## The proxy — what it is, and what it is not

- **Labelled LM-tone proxy.** Scoring every Beige Book's full text with the live LM dictionary is
  the beat-7 extension; here the tone is a **small hardcoded, narrative-anchored reconstruction**
  of that net-tone score — following the desk convention for labelled proxy series in
  **[Study 358 — Watch-Index](../../358-watch-index/)** and
  **[Study 708 — Eurovision-Effect](../../708-eurovision-effect/)**. It is **never** presented
  under a real-tape banner; the release *dates* and *SPY* are real. Because the proxy is our
  construction we cannot certify magnitude — but the study's null is corroborated by the regime
  confound (which the proxy cannot manufacture away) and a synthetic control proving the engine
  recovers a *planted* tone→drift link.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.event_forward_returns`](../beige_book_tone/strategy.py),
  [`strategy.summarize`](../beige_book_tone/strategy.py) (Welch *t* + placebo *p*),
  [`strategy.tone_drift_regression`](../beige_book_tone/strategy.py) (Newey-West HAC dose-response),
  [`strategy.event_overlay`](../beige_book_tone/strategy.py) (long-the-window, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic`](../beige_book_tone/data.py) plants a known tone→drift link; `edge = 0` must
  not manufacture significance, a large `edge` must light up the test.

## Data sources used here

- **Beige-Book release calendar** (real Wednesdays) + **labelled LM-tone proxy** (2011–2024,
  112 releases) + **yfinance SPY** daily total-return adjusted close (2010-10-01 → 2024-12-31),
  cached under `_cache/spy.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 387 — Economic-Surprise-Index](../../387-economic-surprise-index/)** and
  **[Study 385 — Jobless-Claims-Momentum](../../385-jobless-claims-momentum/)**: companion
  "does the macro tell lead the tape?" teardowns on hard data (a constructed surprise index,
  initial claims) — the same event-drift + HAC + placebo + costed-overlay method on numbers
  instead of adjectives.
- **[Study 358 — Watch-Index](../../358-watch-index/)** and
  **[Study 708 — Eurovision-Effect](../../708-eurovision-effect/)**: the desk convention for a
  small, clearly-labelled hardcoded proxy series aligned to a real market tape.
