# References & literature map — Study 567 (Uncertainty-Word-Count)

## The claim, at full strength

- **Loughran & McDonald (2011)**, *"When Is a Liability Not a Liability? Textual Analysis,
  Dictionaries, and 10-Ks."* *Journal of Finance* 66(1). The paper that built the finance-specific
  word lists — including the **Uncertainty** list ("approximately", "contingency", "depend",
  "fluctuate", "indefinite", "may", "risk", "uncertain", ...). It shows the *uncertainty* and
  *weak-modal* word densities in 10-Ks relate to **return volatility** and market reactions around
  the filing — the canonical source of this study's claim and the lexicon it proxies.
- **Campbell, Chen, Dhaliwal, Lu & Steele (2014)**, *"The Information Content of Mandatory Risk
  Factor Disclosures in Corporate Filings."* *Review of Accounting Studies* 19(1). More risk-factor
  / uncertainty language predicts higher subsequent **volatility** and beta — the vol half of the
  claim on the risk-factor section specifically.
- **Kravet & Muslu (2013)**, *"Textual Risk Disclosures and Investors' Risk Perceptions."* *Review
  of Accounting Studies* 18(4). Increases in risk/uncertainty disclosure are followed by higher
  return volatility and trading volume — hedgier text → more perceived (and realised) risk.
- **Bochkay, Chychyla & Nanda (2020)**, *"Dynamics of CEO Disclosure Style."* *The Accounting
  Review*. Documents how managerial hedging/uncertainty language varies and how the market reads it,
  reinforcing that the *residual* (manager-specific) hedging carries information.
- **Baker, Bloom & Davis (2016)**, *"Measuring Economic Policy Uncertainty."* *Quarterly Journal of
  Economics* 131(4). The macro cousin: counting uncertainty words in newspapers builds an index that
  tracks aggregate volatility — the aggregate analogue of this firm-level signal.

## The measure we build

- The LM **Uncertainty** list is one of the orthogonal finance dictionaries (distinct from the
  Positive / Negative *sentiment* lists). This study scores a firm's **uncertainty density** — the
  token share falling in that list — and tests it against **forward realised volatility** (primary)
  and **forward return** (secondary). The honesty pivot is **confounding by trailing volatility**:
  jumpy firms write hedgier text and vol is autocorrelated, so the engine controls for trailing
  realised vol and reports the *residual* uncertainty slope as the headline.

## Neighbours on this bench (the dedup map)

- **[Study 566 — Earnings-Call-Tone](../../566-earnings-call-tone/)** — scores *net emotional tone*
  (positive − negative sentiment) against post-call **drift** (a *return* signal). Study 567 scores a
  *different, orthogonal* lexicon — the LM **Uncertainty**/hedging list — against **forward
  volatility** (a *risk* signal). Different words, different outcome.
- **[Study 259 — News-Tone](../../259-news-tone/)** and **[Study 392 — Glassdoor-Sentiment](../../392-glassdoor-sentiment/)**
  — aggregate mood / employer-sentiment signals, not a firm's own filing scored for hedging density
  against its own forward vol.
- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** / **[Study 335 — Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)**
  — survey / retail-buzz sentiment. Again *sentiment* (bullish/bearish), not the *uncertainty*
  dictionary predicting *vol*.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the plain-minus-hedgy long-short
  spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  uncertainty labels against (forward vol, trailing vol) and refit the controlled slope, reading its
  tail probability.
- **Confound control (omitted-variable / partialling-out)** — the trailing-vol control is a
  Frisch-Waugh partialling of vol persistence out of the uncertainty→forward-vol relation, isolating
  the textual residual.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on a **real** tape for `REAL`; synthetic-only studies cap at `WEAK`), the seed-robust synthetic
  control (≥ 20 seeds), one documented execution lag, and costs one-way × NAV with shorts paying
  borrow.
