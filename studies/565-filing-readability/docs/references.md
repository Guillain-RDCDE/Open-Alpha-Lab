# References & literature map — Study 565 (Filing-Readability)

## The claim, at full strength

- **Loughran & McDonald (2014)**, *"Measuring Readability in Financial Disclosures."* *Journal of
  Finance* 69(4). The canonical statement of the **readability anomaly** and the paper this study
  proxies. Shows that the Gunning **fog** index is a *poor* readability measure in a financial
  setting (its "complex words" are often common finance terms), and that plain **10-K file size** is
  a cleaner, more robust proxy for how hard a filing is to process — one that predicts higher
  post-filing return volatility and (with earnings surprises) the cross-section of returns.
- **Loughran & McDonald (2011)**, *"When Is a Liability Not a Liability? Textual Analysis,
  Dictionaries, and 10-Ks."* *Journal of Finance* 66(1). The foundational LM textual-analysis paper
  and the LM sentiment word lists; establishes that finance-specific text measures beat generic
  ones — the methodological backbone for treating 10-K text as a signal.
- **Li (2008)**, *"Annual Report Readability, Current Earnings, and Earnings Persistence."* *Journal
  of Accounting and Economics* 45(2–3). The seminal *obfuscation* result: firms with **lower
  earnings** file **less readable** (longer, higher-fog) annual reports, and less-readable filings
  have less persistent (worse) earnings — the mechanism behind the return anomaly.
- **You & Zhang (2009)**, *"Financial Reporting Complexity and Investor Underreaction to 10-K
  Information."* *Review of Accounting Studies* 14. The return-drift leg: investors **underreact** to
  the information in longer, more complex 10-Ks, producing stronger post-filing return drift — the
  tradable expression of the anomaly.
- **Cohen, Malloy & Nguyen (2020)**, *"Lazy Prices."* *Journal of Finance* 75(3). A modern
  descendant: **changes** in 10-K/10-Q text (year-over-year similarity) predict returns — murkier /
  more-changed filings precede lower returns. The anomaly extended from the level to the delta.

## The readability legs we build

- The LM literature names three observable readability proxies: the Gunning **fog** index (words per
  sentence + share of "complex" words), raw document **length** (word count), and the 10-K **file
  size** (LM's preferred robust proxy). This study standardises all three into a composite
  *obfuscation* score (higher = less readable), and — following LM — also reports each leg alone,
  where file size and length are the cleaner signals and fog is the noisiest.

## Why this study is synthetic-only (the data-availability limitation)

- A real replication needs 10-K **full text** (SEC EDGAR), parsed to fog/length/file-size, aligned
  **by filing date** with each firm's **forward** return from a **survivorship-free**, point-in-time
  return tape (CRSP/Compustat identifiers). None of that is a free, no-key retail artifact
  (yfinance carries no filing text). The desk rule: `REAL` is earned by a **real** tape at *t* ≥ 2;
  a synthetic-only study is capped at `WEAK`/`NONE`. Same posture as the desk's
  [273 Lego-returns](../../273-lego-returns/), [275 Whisky-cask](../../275-whisky-cask/) and
  [276 Sneaker-resale](../../276-sneaker-resale/) studies, where the free real tape does not exist.

## Neighbours on this bench (the dedup map)

- **[257 AAII-sentiment](../../257-aaii-sentiment/)**, **[335 Buzz-sentiment-ETF](../../335-buzz-sentiment-etf/)**,
  **[392 Glassdoor-sentiment](../../392-glassdoor-sentiment/)** — *sentiment / tone / opinion*
  signals. Study 565 tests the LM **readability / length** anomaly: a *structural* property of the
  filing (how hard it is to read), not its sentiment.
- **[540 Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)** — an accounting/return anomaly on
  a survivor basket; shares the IC / tercile-sort / placebo / synthetic-control machinery and the
  survivorship caveat, but sorts on *distress*, not *readability*.

## Shared method

- **Information coefficient** — the Spearman rank correlation of signal vs forward return, with a
  Fisher-z *t*-stat (`atanh(IC)·√(n−3)`), the standard cross-sectional factor test.
- **Welch (1947)** — the unequal-variance two-sample *t* used for the readable-minus-murky bucket
  spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  readability labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (`REAL` needs a
  robust *t* ≥ 2 on a **real** tape; synthetic control is a machinery proof, never market evidence;
  synthetic-only studies cap at `WEAK`), the survivorship caveat, one execution lag, and costs
  one-way × NAV with shorts paying borrow.
