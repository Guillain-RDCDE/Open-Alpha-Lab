# References & literature map — Study 588 (LLM-Headline-Sentiment)

## The claim, at full strength

- **Lopez-Lira & Tang (2023/2024)**, *"Can ChatGPT Forecast Stock Price Movements? Return
  Predictability and Large Language Models."* SSRN working paper. The paper that lit the fuse: prompt
  an LLM with news headlines, have it label the sentiment, and report next-day return predictability
  and a long-short strategy. The canonical statement of the modern LLM-headline-sentiment claim this
  study demos — and a live example of how such results depend on the timing convention and the
  reporting of the *best* configuration.
- **Tetlock (2007)**, *"Giving Content to Investor Sentiment: The Role of Media in the Stock
  Market."* *Journal of Finance* 62(3). The foundational media-sentiment result: pessimism in a
  daily news column forecasts downward pressure and a subsequent reversal. The pre-LLM ancestor of
  the claim — measured with a word-count (General Inquirer) dictionary.
- **Loughran & McDonald (2011)**, *"When Is a Liability Not a Liability? Textual Analysis,
  Dictionaries, and 10-Ks."* *Journal of Finance* 66(1). The finance-specific sentiment **lexicon**
  the LLM approach claims to upgrade — and the reminder that general-purpose word lists mislabel
  finance text (why context-aware scoring is appealing).
- **García, Hu & Rohrer (2023)**, *"The colour of finance words."* *Journal of Financial Economics*.
  A machine-learning sentiment dictionary estimated directly from returns — the bridge between the
  lexicon era and the LLM era, and a caution on in-sample fitting of sentiment to returns.

## The traps this study isolates

- **Multiple testing / data snooping.**
  - **White (2000)**, *"A Reality Check for Data Snooping."* *Econometrica* 68(5). The bootstrap
    "Reality Check" that tests the *best* model in a family against the joint null — the exact
    correction our `maxt_pvalue` implements for best-of-K sentiment recipes.
  - **Romano & Wolf (2005)** and **Westfall & Young (1993)** — step-down and permutation-max
    family-wise error control; the permutation-max null we use to price the best-of-40 winner.
  - **Harvey, Liu & Zhu (2016)**, *"…and the Cross-Section of Expected Returns."* *Review of
    Financial Studies* 29(1). Why a *t* of ~2 is not enough once you account for the hundreds of
    strategies (or prompts) tried — the multiple-testing reckoning for factor/signal zoos.
- **Look-ahead / hindsight.**
  - The **point-in-time** problem: an LLM scoring historical headlines has (a) training data that
    postdates the trade and (b) a strong temptation to align the score with the session it
    describes. The contemporaneous-vs-lagged contrast (`look_ahead_contrast`) is the standard
    diagnostic; see also the desk's [Study 259 — News-Tone](../../259-news-tone/) for the
    within-month/prior-month version of the same demolition.

## Shared method

- **Newey & West (1987)** — the heteroskedasticity- and autocorrelation-consistent (HAC) standard
  error used on the predictive-regression slope (persistent sentiment ⇒ autocorrelated regressors,
  so the naive OLS *t* is over-confident).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: permute the
  return labels against sentiment and read the tail probability of the observed |t|.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (`REAL` needs a
  robust *t* ≥ 2 on a **real** tape; synthetic-only is capped at `WEAK`), one execution lag
  (today's score trades tomorrow's return), costs one-way × NAV with shorts paying borrow, and the
  seed-robust synthetic positive control.

## Neighbours on this bench (the dedup map)

- **[Study 259 — News-Tone](../../259-news-tone/)** — a curated *daily news-tone* proxy vs the S&P;
  the within-month / prior-month look-ahead demolition. Study 588 is the **LLM-method demo**: the
  best-of-K multiple-testing trap and the HAC-*t* pipeline, on a synthetic tape.
- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** / **[Study 335 —
  Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** — *survey* and *buzz/volume* sentiment as
  predictors. Study 588 is specifically the **LLM-scored-headline** flavour and its overfitting
  traps, not a sentiment *level* or *buzz* study.
- **[Study 566 — Earnings-Call-Tone](../../566-earnings-call-tone/)** — text tone from earnings
  *calls*. Study 588 is the day-ahead *headline* method demo and the multiple-testing correction.
