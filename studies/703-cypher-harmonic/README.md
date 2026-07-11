# Study 703 — Cypher-Harmonic 🔑

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price reverse at a completed Cypher point D? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 5-day fade-at-D returns **+45.77 bps/event, HAC *t* = +0.84** — positive but nowhere near certifiable, with a hit rate (50.4%) dead on a coin flip. Against a drift-matched random-day base rate the Welch *t* is only **+0.81**, and **0 of 7 Bonferroni-corrected tests** (pooled + 6 per-ticker, critical \|*t*\| = 2.69) survive. The point estimate isn't even robust with horizon — it peaks at 5 days and fades by 10. |
| **Tradability** — does it survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Every hold length is weakly positive-to-flat before costs and never clears *t* = 2; net of costs the edge shrinks further and turns outright negative by 20 days (**−32.92 bps at 10 bps**). No horizon is both profitable and certified. |
| **Beats a placebo retracement zone — is 0.786 of XC the pattern's own distinctive signature?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A placebo arm reruns the identical structural pivots with a randomized, off-0.786 D-retracement target. The Cypher's specific ratio **wins on only 3 of 6 tickers** (a coin flip), and at the pooled headline horizon the placebo actually **outperforms** it (Welch *t* = −0.52). |

> **In one sentence:** the Cypher — the only pattern in the harmonic zoo whose D
> point is measured off the freshly-extended **XC** leg rather than XA or AB, with
> C first overshooting the original A swing by 1.13-1.414x — produces a pooled
> 5-day fade that is positive but statistically indistinguishable from noise
> (*t* = 0.84), underperforms a drift-matched base rate at every horizon
> (0/7 Bonferroni-corrected tests survive), loses to a randomized-ratio placebo
> at the pooled level, and turns net-negative by 20 days on six liquid daily
> tapes (2001/2010→2026).

## What we tested

The Cypher harmonic: four confirmed swing pivots X, A, B, C where B retraces XA by
0.382-0.618 (the same shallow band as the Crab), **C overshoots the original A
swing** by 1.13-1.414x of XA (unlike every sibling, where C sits *between* B and
A), and **D retraces the resulting XC leg — not XA, not AB — by exactly 78.6%**,
the pattern's uniquely-referenced signature ratio. On SPY, QQQ, AAPL, MSFT, TSLA
and NVDA daily bars (the identical basket as siblings
[700-bat-harmonic](../700-bat-harmonic/) and
[701-crab-harmonic](../701-crab-harmonic/)), we detect every such candidate off a
percentage-threshold zigzag (confirmed pivots only — no look-ahead), scan forward
for the first touch of the projected D, and measure the forward return of fading
there — against **two** independent controls: a random-day base rate matched to
the same directional mix (Bonferroni-corrected across 7 looks), and a placebo arm
using the identical pivots with an arbitrary, off-0.786 D-retracement. Dedup vs
[468-gartley-harmonic](../468-gartley-harmonic/) (the same 0.786 number, but off
XA — different geometry entirely) and
[699-butterfly-harmonic](../699-butterfly-harmonic/) /
[700-bat-harmonic](../700-bat-harmonic/) /
[701-crab-harmonic](../701-crab-harmonic/) /
[702-shark-harmonic](../702-shark-harmonic/) above — see
[docs/references.md](docs/references.md) for the full dedup map against the rest
of the harmonic-pattern family.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Cypher pattern actually looks like on a chart, why its D point is measured differently from every other harmonic pattern, and what the real tape shows when you actually fade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pivot-confirmation mechanics, the HAC/Welch splits, the Bonferroni correction, the per-instrument breakdown, the cost sweep, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cypher_harmonic/`](cypher_harmonic/). No survivorship — six currently-listed,
individually named large-cap/ETF tickers, not a membership-conditioned panel.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
