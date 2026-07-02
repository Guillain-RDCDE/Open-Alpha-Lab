# Results — Study 588 (LLM-Headline-Sentiment): the method, and its two overfitting traps

*Generated from [`llm_headline_sentiment/`](../llm_headline_sentiment/) on a **deterministic,
offline synthetic tape** (there is no free, dated, point-in-time history of LLM-scored headlines
— see the SIGNAL-axis caveat below). Three synthetic worlds, all seeded at 588 and reproducible
by re-running the package: an **edge** world (a small predictive link planted, series fingerprint
`06461296a1f8`), a **null** world (`802a27d1a3e5`), and a 40-signal **feature bank** for the
multiple-testing demo (`47ff60cb8a0e`). 1,500 business days each. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE` · "Overfitting trap survived?" `BUSTED`

The 2020s pitch: point an LLM at the day's news headlines, have it score the mood, and use that
score to forecast *tomorrow's* market — the context-aware upgrade of Loughran-McDonald lexicon
sentiment. This study is a **method demo**: we build the honest predictive-regression pipeline (a
Newey-West HAC *t*, a label-shuffle placebo), prove it works on a planted synthetic edge, and then
show the two ways this exact kind of study fools people — **multiple testing** (try many prompts,
report the winner) and **look-ahead** (score headlines with hindsight).

**Signal `WEAK` is a ceiling, not a measurement.** There is no honest free real tape of dated,
point-in-time LLM headline scores (the headlines are paywalled and scoring them as-of each
historical day without leaking the model's training future is the hard part). A `REAL` stamp
requires a robust *t* ≥ 2 on a **real** tape; this study has none, so it is capped at `WEAK` by
construction — exactly like the desk's lego-returns / whisky-cask / sneaker-resale synthetic-only
studies. Everything below is either a **machinery proof** (the engine detects a planted effect) or
a **cautionary demo** (how the machinery is abused). Neither is market evidence.

## The engine is faithful — the planted-edge world

On the synthetic **edge** world (a small link `forward_ret ≈ β·sentiment + noise`, β chosen so the
edge is realistically tiny), the honest lagged pipeline recovers it:

| | value |
|---|---|
| Slope (forward return on sentiment) | **0.0031** |
| **Newey-West HAC *t*** | **+6.33** |
| OLS *t* (no HAC) | +6.39 |
| R² (variance of tomorrow's return explained) | **2.7%** |
| Label-shuffle placebo *p* (2000 perms) | **0.0005** |

R² of ~3% is the right order of magnitude for a *real* headline edge — small, but genuinely in the
tail of the placebo. This is the **positive control**: the machine can bank a real link. (Synthetic
— never cited for the Signal stamp.)

## The engine stays flat at the null

On the **null** world (β = 0, sentiment is pure noise against the future):

| | value |
|---|---|
| Slope | 0.0009 |
| HAC *t* | **+1.79** (below the *t* = 2 bar) |
| Placebo *p* | 0.071 |
| Sign-timer net Sharpe | 0.92 |

No false signal — the detector reads ~noise where there is nothing to find.

## Trap 1 — multiple testing (the money shot)

A researcher rarely tests one sentiment recipe; they try many prompts × models × aggregations and
**report the best**. We build a bank of **40 pure-noise** sentiment features (none predicts) against
one return series:

| | value |
|---|---|
| Best-of-40 |HAC *t*| (the winner, `sent_19`) | **2.90** |
| Naive single-test *p* the miner would quote | **0.007** ✅ "significant!" |
| Honest **max-*t*** family-wise *p* (Westfall-Young / White Reality Check, 1000 perms) | **0.173** ❌ not significant |

The winner's naive *p* = 0.007 **looks** publishable; the permutation-max null — the distribution of
the *best* |t| under no signal — puts it at *p* = 0.173. Repeated over 25 independent pure-noise
banks:

| Test | False-positive rate at α = 0.05 |
|---|---|
| **Naive** "report the winner" | **84%** |
| **Max-*t*** correction | **0%** |

Try 40 recipes on noise and you find a "significant" one **84% of the time**. Correct for it and the
rate collapses to zero. This is the trap the whole study exists to make legible.

## Trap 2 — look-ahead / hindsight labelling

Score the headlines with a mood label built *after the fact* (or a vendor feed timestamped to the
session it describes), and the contemporaneous fit balloons:

| Design | HAC *t* |
|---|---|
| **Lagged (honest):** today's mood → tomorrow's return | **+6.33** |
| **Contemporaneous (leak):** hindsight-tainted mood → same-day return | **+16.1** |

The contemporaneous |t| is ~2.5× the tradable one — a mood score that partly *reads the answer* is
not a forecast. The gap between the two is the tell.

## The tradable next-day sign timer — on the planted edge

Even where a real edge exists, a one-day sign timer flips constantly:

| | value |
|---|---|
| Gross annualised Sharpe | **2.70** |
| Net Sharpe (1 bp/flip one-way + 50 bps/yr borrow) | **2.58** |
| Daily turnover (one-way) | **0.77** |

The Sharpe *survives* costs here **only because the edge was planted at full strength in a
frictionless synthetic world**. On the null world the same timer nets Sharpe 0.92 (noise). Neither
is a claim about markets.

## Synthetic positive control — seed-robust (25 seeds)

| Planted β | Mean HAC *t* (25 seeds) | |
|---|---|---|
| 0.0000 (null) | **+0.09** | flat — no false signal |
| 0.0008 | +1.79 | below the bar |
| 0.0015 | +3.27 | clears the bar |
| 0.0022 (headline) | +4.76 | edge visible |
| 0.0030 | +6.46 | strong |
| 0.0045 | +9.65 | very strong |

At the null the mean HAC *t* is ≈ 0; planting a genuine link drives it up monotonically and past 2.
The detector works — so the traps above are about *misuse*, not a broken engine. (Control only.)

## Why this can never be `REAL` here

1. **No real tape.** Dated, point-in-time LLM headline scores are not freely reachable — the
   headlines are licensed text, and scoring them as-of each historical day without leaking the
   model's own training future is the crux. Synthetic-only ⇒ capped at `WEAK`.
2. **The published edges are small and fragile.** Where LLM-sentiment edges are reported, R² is a
   few percent (as planted here) and decays fast as the method is arbitraged and as models change.
3. **The literature's headline numbers are the *un*-corrected ones.** Much of the "LLM beats the
   market" excitement is exactly Trap 1 (best prompt reported) and Trap 2 (contemporaneous scoring)
   — the two failures this demo isolates.

## The honest takeaway

The method is real and the pipeline works (planted edge → HAC *t* +6.33, placebo *p* 0.0005). But on
a synthetic-only footing it can never earn more than `WEAK`, and the demo's whole point is the two
traps: try 40 recipes on **noise** and you'll "find" a winner **84%** of the time (max-*t* correction
→ **0%**), and hindsight-labelled mood inflates a *t* of 6.3 to 16.1. `WEAK` × `MIRAGE`, overfitting
trap `BUSTED`.
