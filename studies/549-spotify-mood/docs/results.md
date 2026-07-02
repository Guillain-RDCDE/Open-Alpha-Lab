# Results — Study 549 (Spotify-Mood): does streaming valence lead the market?

*Generated from [`spotify_mood/`](../spotify_mood/). The **market** half is real: monthly ``^GSPC``
(S&P 500) returns from this study's cached yfinance tape, **2010-02 → 2026-05** (196 months;
partial June 2026 dropped), market-return fingerprint `431d740a90aa`. The **mood** half is
**synthetic** (a seeded AR(1) valence proxy with **no** planted edge — a plausible-but-unprivileged
honest mood series), because a free, survivorship-clean historical monthly valence panel does not
exist (Spotify closed the audio-features API to new apps in Nov 2024). Joined panel fingerprint
`1e087c2e441e`. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Music sentiment on the tape?" `UNTESTABLE (no real valence tape)`

The music-sentiment literature (Edmans, Fernandez-Perez, Garel & Indriawan 2022) argues the
aggregate musical **valence** of what people stream is a mood proxy that co-moves with — and might
*lead* — the stock market. We build the honest retail version of the test and immediately hit the
wall that decides the SIGNAL axis: **there is no free real valence tape.** Spotify's audio-features
endpoint (the only public source of ``valence``) was **closed to new applications in November
2024**, and even before that never exposed a clean historical monthly panel of the global top-chart
valence. Edmans et al. used a licensed proprietary dataset. So our mood series is **synthetic** —
and a synthetic mood input can *never* certify a REAL signal (that needs a robust *t* on a **real**
valence tape). `NONE`, capped by data availability.

To show the machinery is nonetheless faithful, we run the honest test with a *plausible-but-null*
synthetic mood proxy against the real market: **lag-1 HAC *t* = −1.25** (not significant, and the
*wrong* sign for the claim), **placebo *p* = 0.231** (the shuffle null swallows it), directional
**hit-rate 46.7%** versus a **66.2%** unconditional base rate (a *−19-point* miss), and a lag-1..5
sweep whose only sub-|2| exception is a data-mined lag-2 (*t* −2.11, still the wrong sign, below the
Bonferroni bar ~2.58). Tradability is `MIRAGE`: the long-only mood-timing rule earns **+4.7%/yr**
against the market's **+13.2%/yr** over the same live months — it sits out most of the bull run —
and the long-short variant is **negative** (−3.3%/yr gross).

## Data stamp

| Source | Window | n | Fingerprint |
|---|---|---|---|
| ^GSPC monthly returns (real, yfinance, auto-adjust) | 2010-02 → 2026-05 | 196 months | `431d740a90aa` |
| Synthetic valence proxy (seed 549, `predictive_beta`=0, joined to real months) | 2010-02 → 2026-05 | 196 | (panel `1e087c2e441e`) |

## The headline predictive regression — lag-1, HAC

Regress *next*-month ^GSPC return on *this*-month valence (z-scored), Newey-West HAC se.

| | value |
|---|---|
| Slope (return per +1σ valence) | **−0.0032** |
| Plain OLS *t* | −1.06 |
| **Newey-West HAC *t*** | **−1.25** |
| R² | 0.006 |
| n | 195 |

The slope is small, statistically indistinguishable from zero, and — for the claim (happier
→ higher next return) — the *wrong sign*. The bar for a REAL signal (HAC |*t*| ≥ 2 on a **real**
valence tape) is not met, and could not be met here even in principle: this is a synthetic mood
input.

## The placebo null

| | value |
|---|---|
| Observed \|slope\| | 0.0032 |
| Circular-shift placebo *p* (2000 shifts) | **0.231** |

Roughly a quarter of random valence↔return alignments produce a slope at least this large — the
observed relation sits comfortably inside the noise band.

## The lag-sweep — the Granger multiple-comparisons trap

| Lag (months) | Slope | HAC *t* |
|---|---|---|
| 1 | −0.003 | **−1.25** |
| 2 | −0.005 | **−2.11** |
| 3 | −0.003 | −1.16 |
| 4 | −0.004 | −1.77 |
| 5 | −0.004 | −1.73 |

Five lags tested; the largest \|HAC *t*\| is **2.11** at lag 2 — below a Bonferroni bar of ~2.58 for
5 comparisons, and *negative* (the opposite of the claim). Cherry-picking the best lag is exactly
the trap the sweep exposes: there is no robust lead-lag here.

## Directional hit-rate vs the honest base rate

| | value |
|---|---|
| "valence-up predicts market-up" hit-rate | **46.7%** |
| Unconditional base rate (bigger of up/down share) | **66.2%** |

The mood-direction call is *worse* than a coin — and far worse than just always predicting the
majority outcome (up).

## Tradability — the mood-timing rule (one-month execution lag)

| Rule | Live months | Gross ann. | Net ann. | vs buy-&-hold market |
|---|---|---|---|---|
| Long-only (long when last month's valence > trailing median, else cash) | 73 | **+4.7%** | **+4.6%** | market **+13.2%/yr** over same months |
| Long-short (short in low-mood months, pays 100 bps borrow) | 183 | **−3.3%** | **−4.0%** | market **+13.2%/yr** |

The long-only rule *underperforms* simple buy-and-hold by ~8 pts/yr (it sits in cash during good
months it can't foresee); the long-short variant is outright negative. Costs (5 bps/leg one-way ×
NAV) barely move it because there is no edge to erode. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `predictive_beta` | Mean lag-1 HAC *t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.03** | flat — no false signal |
| 0.005 | +1.54 | edge emerging |
| 0.010 | +3.11 | clears the bar |
| 0.015 | +4.68 | strong |
| 0.020 | +6.25 | very strong |

At the null the HAC *t* is ≈ 0; planting a genuine predictive edge drives it positive and past +2
by `predictive_beta` ≈ 0.007. So the detector works — the flat real-tape reading is the *data*
talking (an honest, unprivileged mood series carries no market-timing information), not a broken
engine. (Control only; never cited for a real-tape stamp — and there is no real valence tape to
stamp.)

## Why this can't certify

1. **No real valence tape (the decisive limitation).** The only public source of Spotify
   ``valence`` — the audio-features API — was closed to new applications in November 2024, and never
   exposed a clean survivorship-free historical monthly chart-valence panel. The published
   music-sentiment result relied on a licensed proprietary dataset. A synthetic mood input can
   never clear the REAL bar; the study is honest about this on the SIGNAL axis.
2. **Alt-data survivorship & construction risk.** Even with data, "top-streamed songs" is a moving,
   platform-defined, survivorship-laden universe; charts, catalog availability and Spotify's own
   valence model all changed over the window. The mood proxy is fragile by construction.
3. **Power.** 196 monthly observations is thin for a 1–3-month lead-lag; the honest inference bar
   (HAC + placebo + Bonferroni over lags) is exactly what keeps a lucky lag from being oversold.

## The honest takeaway

"The mood of the songs people stream leads the market" is a charming alt-data story, and the
academic version (on a licensed valence tape) reports a contemporaneous mood effect. But on a free
retail stack the story cannot even be *tested*: there is no real historical valence tape. Running
the honest test with a plausible synthetic mood proxy against the real S&P returns a wrong-signed,
insignificant lag-1 HAC *t* (−1.25), a placebo *p* of 0.23, a below-coin hit-rate, and a mood-timing
rule that trails buy-and-hold by ~8 pts/yr. `NONE` × `MIRAGE`. The synthetic control confirms the
engine *would* catch a real edge — so the verdict is about the (missing) data, not the code.
