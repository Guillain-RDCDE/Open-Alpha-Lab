# Study 701 — Crab-Harmonic 🦀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price reverse at a completed Crab point D? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 5-day fade-at-D returns **−82.70 bps/event, HAC *t* = −1.30** — negative, with a hit rate (39.5%) *below* a coin flip. Against a drift-matched random-day base rate the Welch *t* is **−0.61** (Crab underperforms a matched-direction random entry), and **0 of 7 Bonferroni-corrected tests** (pooled + 6 per-ticker, critical \|*t*\| = 2.69) survive. At 10 days the fade turns individually *significant in the losing direction* (HAC *t* = −2.40). |
| **Tradability** — does it survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Every hold length (1/3/5/10/20 days) is negative on average, gross and net; costs only compound the loss; by 20 days the fade loses **503 bps gross**, a statistically significant loss (*t* = −2.76). |
| **Beats a placebo extension zone — is 1.618 the "sharpest, most exact" ratio, as Carney claims?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A placebo arm reruns the identical structural pivots with a randomized, off-1.618 D-target. The Crab's specific ratio **wins on only 3 of 6 tickers** (a coin flip) and shows no pooled edge (Welch *t* = +0.28). |

> **In one sentence:** the Crab harmonic — Scott Carney's self-declared "sharpest,
> most exact" pattern, a single 1.618 extension past the original X leg — does not
> just fail to find a reversal on six liquid daily tapes (2001/2010→2026), it loses
> money: the fade is negative at every horizon, turns significantly negative by 10
> days (the tape *continues* rather than reverses), fails a Bonferroni-corrected
> comparison against a drift-matched base rate 7/7 times, and does not beat a placebo
> control built from the identical pivots with an arbitrary, non-1.618 target.

## What we tested

The Crab harmonic: five confirmed swing pivots X, A, B, C where B retraces XA by
0.382-0.618 (shallower than the Butterfly's fixed 0.786, close to the Bat's
0.382-0.50), C retraces AB by 0.382-0.886 (the shared XABCD band), and **D extends the
*original* XA leg by exactly 1.618x, past point X** — the single sharpest, most extreme
target in the harmonic zoo, unlike the Butterfly's 1.27-1.618 *range*. On SPY, QQQ,
AAPL, MSFT, TSLA and NVDA daily bars (the identical basket as siblings
[699-butterfly-harmonic](../699-butterfly-harmonic/) and
[700-bat-harmonic](../700-bat-harmonic/)), we detect every such candidate off a
percentage-threshold zigzag (confirmed pivots only — no look-ahead), scan forward for
the first touch of the projected D, and measure the forward return of fading there —
against **two** independent controls: a random-day base rate matched to the same
directional mix (Bonferroni-corrected across 7 looks), and a placebo arm using the
identical pivots with an arbitrary, off-1.618 D-target. Dedup vs
[468-gartley-harmonic](../468-gartley-harmonic/) (D stays *inside* [X, A], different B
ratio) and [699-butterfly-harmonic](../699-butterfly-harmonic/) /
[700-bat-harmonic](../700-bat-harmonic/) above — see
[docs/references.md](docs/references.md) for the full dedup map against the rest of
the harmonic-pattern family.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Crab pattern actually looks like on a chart, why Carney says it's the "sharpest" harmonic, and what the real tape shows when you actually fade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pivot-confirmation mechanics, the HAC/Welch splits, the Bonferroni correction, the per-instrument breakdown, the cost sweep, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`crab_harmonic/`](crab_harmonic/). No survivorship — six currently-listed,
individually named large-cap/ETF tickers, not a membership-conditioned panel.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
