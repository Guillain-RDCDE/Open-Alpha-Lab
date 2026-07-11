# Study 702 — Shark-Harmonic 🦈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price reverse in the Shark's 0.886-1.13 completion zone? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 5-day fade-at-D returns **+22.81 bps/event, HAC *t* = +0.42** — small, positive, nowhere near certified, hit rate (54.3%) straddling a coin flip (Wilson [44.2%, 64.0%]). Against a drift-matched random-day base rate the Welch *t* is **+0.60**, and **0 of 7 Bonferroni-corrected tests** (pooled + 6 per-ticker, critical \|*t*\| = 2.69) survive — the one nominally-significant split (SPY, *t* = +2.57) rests on just 11 events. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No hold length (1/3/5/10/20 days) ever clears \|*t*\| = 2, gross or net; the point estimate isn't even monotone in horizon (net-of-costs dips negative at 10 days, then drifts positive again at 20) — the signature of noise, not a decaying real edge. |
| **Beats a placebo completion zone — is 0.886-1.13 the Shark's defining "5-0" edge, as Carney claims?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A placebo arm reruns the identical structural pivots with a D-zone re-centered away from 0.886-1.13. The Shark's specific zone **wins on only 3 of 6 tickers** (a coin flip) and shows no pooled edge (Welch *t* = +0.26). |

> **In one sentence:** the Shark harmonic — the only member of the zoo built entirely from
> *extensions* (B overshoots 1.13-1.618x past X, C overshoots a further 1.618-2.24x past A,
> never a single retracement leg) completing into a genuine 0.886-1.13 price ZONE rather than a
> point target — shows no reversal edge on six liquid daily tapes (2001/2010→2026): the fade is
> statistically indistinguishable from noise, fails a Bonferroni-corrected comparison against a
> drift-matched base rate 7/7 times, and does not beat a placebo control built from the
> identical pivots with an arbitrary, off-band completion zone.

## What we tested

The Shark ("5-0") harmonic: five confirmed swing pivots X, A, B, C where B **extends** XA by
1.13-1.618x past point X (not a retracement — the Shark's first break from the rest of the
zoo), C **extends** AB by a further 1.618-2.24x past point B, and the completion point D is a
**price ZONE at 0.886-1.13x the original XA leg** — a range, unlike every other pattern's
single-point target. On SPY, QQQ, AAPL, MSFT, TSLA and NVDA daily bars (the identical basket as
siblings [699-butterfly-harmonic](../699-butterfly-harmonic/),
[700-bat-harmonic](../700-bat-harmonic/), [701-crab-harmonic](../701-crab-harmonic/) and
[703-cypher-harmonic](../703-cypher-harmonic/)), we detect every such candidate off a
percentage-threshold zigzag (confirmed pivots only — no look-ahead), scan forward for the first
overlap with the projected D-zone, and measure the forward return of fading there — against
**two** independent controls: a random-day base rate matched to the same directional mix
(Bonferroni-corrected across 7 looks), and a placebo arm using the identical pivots with a
re-centered, off-band D-zone. Dedup vs [468-gartley-harmonic](../468-gartley-harmonic/) (every
leg a retracement, D never leaves [X, A]) and the four extension/overshoot siblings above — see
[docs/references.md](docs/references.md) for the full dedup map against the rest of the
harmonic-pattern family.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Shark pattern actually looks like on a chart, why it's the "odd one out" of the harmonic zoo, and what the real tape shows when you actually fade its completion zone |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pivot-confirmation mechanics, the HAC/Welch splits, the Bonferroni correction, the per-instrument breakdown, the cost sweep, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`shark_harmonic/`](shark_harmonic/). No survivorship — six currently-listed,
individually named large-cap/ETF tickers, not a membership-conditioned panel.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
