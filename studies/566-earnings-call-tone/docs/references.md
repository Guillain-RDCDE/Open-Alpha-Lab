# References & literature map — Study 566 (Earnings-Call-Tone)

## The claim, at full strength

- **Loughran & McDonald (2011)**, *"When Is a Liability Not a Liability? Textual Analysis,
  Dictionaries, and 10-Ks."* *Journal of Finance* 66(1). The foundational finance sentiment lexicon:
  generic (Harvard) word lists mis-classify finance text, and a domain-specific positive/negative
  dictionary is what makes textual *tone* informative for returns. The measure our `net_tone` proxies
  ((positive − negative) share under a finance lexicon).
- **Price, Doran, Peterson & Bliss (2012)**, *"Earnings Conference Calls and Stock Returns: The
  Incremental Informativeness of Textual Tone."* *Journal of Banking & Finance* 36(4). The direct
  statement of this study's claim: the *linguistic tone* of an earnings **call** predicts returns and
  drift *incrementally* to the earnings surprise — i.e. tone matters **after controlling for the
  number**. The paper this study operationalises.
- **Mayew & Venkatachalam (2012)**, *"The Power of Voice: Managerial Affective States and Future Firm
  Performance."* *Journal of Finance* 67(1). Vocal/affective cues on the call carry information about
  future performance beyond the transcript — evidence the *tone* channel is real, not just a repackaged
  number.
- **Davis, Piger & Sedor (2012)** / **Davis, Ge, Matsumoto & Zhang (2015)** — optimistic vs pessimistic
  language in earnings press releases and calls moves markets and predicts future performance; the
  broader body of "language of disclosure" evidence the claim sits inside.

## The confound this study foregrounds — PEAD

- **Bernard & Thomas (1989, 1990)**, *"Post-Earnings-Announcement Drift"* — the numeric surprise (SUE)
  drifts for weeks after the announcement. Because call tone is *correlated with* the surprise (good
  quarters produce upbeat calls), a naive tone→return regression double-counts PEAD. The honest
  linguistic test must control for the surprise — the pivot of this study, and why the naive slope is
  an upper bound. See the desk's own **[Study 363 — PEAD-drift](../../363-pead-drift/)** and
  **[Study 534 — revenue-surprise-drift](../../534-revenue-surprise-drift/)** for the numeric drifts
  this rides on top of.

## Neighbours on this bench (the dedup map)

- **[Study 363 — PEAD-drift](../../363-pead-drift/)** / **[534 — revenue-surprise-drift](../../534-revenue-surprise-drift/)**
  — drift on the *number* (the earnings / revenue surprise). Study 566 is the *linguistic* cousin:
  drift on the *words*, controlling for the number.
- **[Study 259 — news-tone](../../259-news-tone/)** / **[300 — sports-sentiment](../../300-sports-sentiment/)**
  / **[392 — glassdoor-sentiment](../../392-glassdoor-sentiment/)** — *aggregate / employer / macro* mood
  scores against broad returns. Study 566 is a firm-level **event** study: one transcript's tone against
  that firm's own post-call CAR.
- **[Study 257 — aaii-sentiment](../../257-aaii-sentiment/)** / **[261 — put-call-ratio](../../261-put-call-ratio/)**
  — market-wide sentiment gauges, not transcript language.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the upbeat − guarded tone-tail spread.
- **Multiple regression / partialling-out** (Frisch-Waugh-Lovell) — controlling for the numeric surprise
  to isolate the tone coefficient; the difference between the naive and controlled slope *is* the confound.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the tone
  labels against (CAR, surprise) and read the controlled-slope tail probability.
- **Data-availability limitation** — as with the desk's synthetic-only studies
  ([273 lego-returns](../../273-lego-returns/), [275 whisky-cask](../../275-whisky-cask/),
  [276 sneaker-resale](../../276-sneaker-resale/)), no free real feed exists, so the Signal axis is
  capped at `WEAK` and the limitation is named openly.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (`REAL` needs a
  robust *t* ≥ 2 on a *real* tape; a synthetic control is a machinery proof, never market evidence), one
  execution lag, costs one-way × NAV with shorts paying borrow, and no silent caps.
