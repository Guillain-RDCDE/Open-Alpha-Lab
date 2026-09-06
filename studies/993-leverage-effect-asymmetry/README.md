# Study 993 — Down Hurts More ⚖

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the volatility response to returns genuinely asymmetric? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | For SPY over 7,764 sessions, volatility over the five days after a down day averaged **15.9%** against **13.5%** after an up day — a ratio of **1.18×**. That survives the obvious objection: matching on the *size* of the move (down days are bigger on average, and big moves are followed by volatility whatever their sign) leaves a ratio of **1.15×**. It survives honest inference too — block-bootstrapping the difference gives *t* = **+7.70** against the naive +9.37. And the parametric version agrees: EGARCH gamma = **-0.110**, implying a down shock raises volatility 1.25× as much as an equal up shock. The news-impact curve bottoms out at **z = +0.28**, shifted toward positive returns exactly as the leverage story predicts. |
| **Tradability** — is the asymmetry big enough to change a hedge? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Now the part the name gets wrong. If financial leverage were the mechanism, assets with **no balance sheet** could not show the effect. Gold's ratio is 1.02× and Bitcoin's is 1.03×, against equities' 1.18× — and the effect is materially weaker there, which is at least consistent with the leverage story. The lead-lag test points the same way: the correlation between returns and *subsequent* volatility changes averages -0.051, against +0.015 for volatility leading returns — leaning **leverage**. For a hedger the practical content is the ratio itself: a 1.18× asymmetric response is why put skew exists and why a delta-hedged short-vol book bleeds asymmetrically. The name is wrong; the effect is real and it is priced. |

> **In one sentence:** Volatility after a down day runs 1.18× that after an up day and the effect survives every control — but it is 1.02× in gold, which has no debt, so whatever causes it, it is not leverage.

## What we tested

Black noticed it in 1976: volatility rises more after a price fall than after an
equal rise. He proposed a mechanism — a falling price raises debt-to-equity, making the residual
equity riskier — and the name **"leverage effect"** stuck. The effect is among the most robust
findings in empirical finance. The explanation has been in doubt for forty years.

This study measures the asymmetry four ways (a forward sign split, an EGARCH gamma, the
news-impact curve's vertex, and the return/volatility-change correlation) and controls for the
three things that manufacture it out of nothing. **Down days are bigger than up days**, and big
moves are followed by volatility whatever their sign — so the split is repeated *matched on
|return|*. **Splitting on the sign of a regressor** is a false-positive machine — so the
difference is block-bootstrapped with the split re-derived inside each resample. **Volatility is
measured with noise** — so the comparison uses forward volatility, never the window containing
the move itself.

Then the part that decides the mechanism, and it needs no econometrics at all: **gold and
Bitcoin have no balance sheets.** There is no debt-to-equity ratio for a falling price to
change. If the asymmetry shows up there, the name is simply wrong. The lead-lag test adds the
other half of the case — leverage requires the return to move *first*, while volatility feedback
(Campbell & Hentschel 1992) requires the volatility change to — and a synthetic EGARCH world
with a **planted gamma** grades every estimator against a known truth and measures how often the
naive test cries wolf.
**Dedup:** distinct from **256-volatility-clustering** and **992-vol-clustering-halflife**
(persistence, not asymmetry), **989-altcoin-downside-beta** (asymmetric *beta* between two
assets, not the volatility response within one), **371-vix-term-structure** and
**445-volatility-skew** (the implied-vol surface this effect helps explain) and
**311-fat-tails** (the unconditional distribution).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | whether falls really do stir up more volatility than rises, and the one comparison that shows the textbook explanation cannot be right |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | magnitude-matched splits, block-bootstrapped differences, news-impact curves and their fitted vertex, an EGARCH gamma, the lead-lag test that separates leverage from volatility feedback, and a planted-gamma control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`downhurts/`](downhurts/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
