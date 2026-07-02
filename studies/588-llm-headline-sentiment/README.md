# Study 588 — LLM-Headline-Sentiment 🤖📰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — can LLM headline mood forecast tomorrow? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | **Method demo, synthetic-only.** No free, dated, point-in-time history of LLM-scored headlines exists (licensed text + the as-of scoring problem), so a `REAL` stamp — which needs a robust *t* ≥ 2 on a **real** tape — is **impossible here** and the axis is capped at `WEAK`. On the planted-edge synthetic world the honest pipeline recovers it cleanly (HAC *t* **+6.33**, R² **2.7%**, placebo *p* **0.0005**) and stays flat at the null (HAC *t* **+1.79**) — a machinery proof, never market evidence. |
| **Tradability** — does the day-ahead timer pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only Sharpe that survives costs (gross **2.70** → net **2.58**) is a **planted** edge in a frictionless synthetic world; the honest lessons are the traps below. A real one-day sign timer flips constantly (turnover **0.77**/day), there is no real tape, and most published "LLM alpha" is the *un*-corrected best-of-K. Nothing bankable. |
| **Overfitting trap survived?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Try **40 pure-noise** sentiment recipes and report the winner: naive single-test *p* = **0.007** ("significant!") — but the family-wise **max-*t*** *p* = **0.173**. Over 25 noise banks the naive false-positive rate is **84%**; the max-*t* correction drops it to **0%**. And a hindsight-labelled score inflates a *t* of **6.3** to **16.1**. Both traps `BUSTED`. |

> **In one sentence:** letting an LLM read the headlines and score the mood is a real *method* — the honest pipeline banks a planted day-ahead edge at HAC *t* +6.33 — but with no free real tape it can never clear `WEAK`, and the study's whole point is the two ways this exact research fools you: try 40 sentiment recipes on **noise** and you'll crown a "significant" winner **84%** of the time (a proper max-*t* correction → **0%**), while scoring headlines with hindsight turns a *t* of 6.3 into 16.1.

## What we tested

The 2020s pitch — the context-aware upgrade of Loughran-McDonald lexicon sentiment: point a language
model at the day's news headlines, have it emit a mood score, and forecast *tomorrow's* return
(Lopez-Lira & Tang 2023; Tetlock 2007). Because no honest free tape of dated, point-in-time LLM
headline scores exists, this is a **method demo** built on a deterministic synthetic world
(`data.py`, seed 588). We build the honest pipeline — a predictive regression with a **Newey-West
HAC *t*** and a **label-shuffle placebo** — prove it recovers a planted edge and stays flat at the
null, then isolate the two failures that produce most "LLM beats the market" headlines: **multiple
testing** (a 40-signal noise bank, naive best-of-K vs a **Westfall-Young / White Reality Check
max-*t*** correction) and **look-ahead** (honest lagged scoring vs a hindsight-tainted
contemporaneous fit). A one-day sign timer with costs + borrow and a seed-robust (25-seed) synthetic
positive control round it out. *Distinct from [259 News-Tone](../259-news-tone/) (curated daily-tone
proxy, within-month placebo), [257 AAII](../257-aaii-sentiment/) / [335 Buzz](../335-buzz-sentiment-etf/)
(survey / buzz levels), and [566 Earnings-Call-Tone](../566-earnings-call-tone/) (call transcripts) —
this is the **LLM-headline method and its overfitting traps**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what LLM headline sentiment is, why it *can* work on a clean world, and the two ways it fools you — the "report the best prompt" trap and the hindsight trap — in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC predictive regression, the label-shuffle placebo, the best-of-K max-*t* (Reality-Check) correction, the lagged-vs-contemporaneous look-ahead contrast, the costed sign timer, and the seed-robust synthetic positive control |

The fingerprinted headline run (edge world fp `06461296a1f8`, null `802a27d1a3e5`, feature bank
`47ff60cb8a0e`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
runs on the deterministic synthetic world in [`llm_headline_sentiment/data.py`](llm_headline_sentiment/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`llm_headline_sentiment/`](llm_headline_sentiment/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
