# References & literature map — Study 543 (Western-Zodiac-CEO)

## The claim, at folklore strength

- **Astro-finance / horoscope investing (popular folklore).** The recurring media claim that a
  CEO's *sun sign* predicts corporate or share-price performance — the horoscope column applied to
  the C-suite. It surfaces perennially in lifestyle finance pieces (e.g. "the best zodiac signs
  for CEOs / for investing") and is the western-astrology cousin of the Chinese *zodiac-year*
  folklore. There is no proposed mechanism; a birthday is independent of how a firm trades. We
  test it generously and show it evaporates.
- **Kolb & Rodriguez (1987)**, *"Friday the Thirteenth: 'Part VII' — A Note."* *Journal of
  Finance* 42(5). A canonical example of the desk's genre: a superstitious calendar/astro pattern
  tested rigorously and found to be a small-sample artifact. The template for treating folklore as
  a disprovable hypothesis, not a strategy.

## Why a CEO-sign study can only ever be WEAK/NONE

- **No mechanism + a fixed label.** A sun sign is set at birth, decades before any price, and does
  not change over a CEO's tenure — so there is no time series to average, only one cross-section of
  a few dozen names. This is the structural reason the study is capped below `REAL` on the SIGNAL
  axis regardless of what the tape happens to print.
- **Multiple comparisons on tiny cells.** Twelve signs over ~30 names leaves 2-3 names per sign —
  hopelessly underpowered for a 12-way test, exactly as **Study 165 — Chinese-Zodiac**'s "12
  animals × ~3 years" is. Picking "the best sign" after the fact is a selected contrast whose
  naive *t* is upward-biased; the honest null is a **max-statistic label-shuffle** (below).

## Neighbours on this bench (the dedup map)

- **[Study 165 — Chinese-Zodiac](../../165-chinese-zodiac/)** — the direct sibling: the *zodiac
  year* (Dragon-year bullishness, pre-CNY rally) on Chinese/Asian equities, with a 12-way
  Bonferroni correction and a power analysis. Study 543 is the **western sun-sign of the CEO**,
  a *cross-sectional* firm-level sort rather than a calendar-year cycle.
- **[Study 84 — Moon-Math](../../84-moon-math/)** and the desk's other astro/calendar folklore —
  celestial-cycle claims tested and dismissed. Study 543 targets the *person* (the CEO's sign),
  not a calendar.

## Shared method

- **One-way ANOVA (Fisher).** The omnibus F across the twelve signs — "do the sign means differ
  more than chance?" On tiny cells the analytic F-p is unreliable, so the reported inference is
  the permutation p.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  sign labels against forward returns and read the F's tail probability, plus a **max-statistic**
  null on the best-vs-rest spread (re-selecting the best sign on every shuffle) to correct for the
  post-hoc selection of the winning sign.
- **Welch (1947)** — the unequal-variance two-sample *t* for the best-sign-vs-rest contrast.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (`REAL`
  needs a robust *t* ≥ 2 on the real tape; literature/curated tables cap at `WEAK`/`NONE`), the
  seed-robust synthetic positive control (≥ 20 seeds), one documented execution lag, and costs
  one-way × NAV with shorts paying borrow.
