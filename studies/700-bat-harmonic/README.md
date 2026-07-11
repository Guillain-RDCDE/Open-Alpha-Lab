# Study 700 — Bat-Harmonic

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price reverse at a completed Bat point D? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 5-day fade-at-D returns **+7.58 bps/event, HAC *t* = +0.17** — a hit rate (40.4%) *below* a coin flip. Against a drift-matched random-day base rate the Welch *t* is +0.45, and **0 of 7 Bonferroni-corrected tests** (pooled + 6 per-ticker, critical \|*t*\| = 2.69) survive. TSLA's individually notable cell (gross *t* = +2.34) is the closest call and still falls short before correction. |
| **Tradability** — does it survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No hold length (1/3/5/10/20 days) clears *t* ≥ 2 on the positive side, and the best gross case (+7.58 bps, 5-day) turns **negative at a single 5 bps one-way cost**. Longer holds bleed more — the 20-day fade loses 143 bps gross on average. |
| **Beats a placebo retracement zone — is 0.886 "the most reliable" ratio, as Carney claims?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A placebo arm reruns the identical structural pivots with a randomized, off-0.886 D-target. The Bat's specific ratio **wins on only 2 of 6 tickers** and shows no pooled edge (Welch *t* = +0.22). |

> **In one sentence:** the Bat harmonic — Scott Carney's self-declared "most reliable" pattern,
> a 0.886 retracement of the original XA leg that stays inside [X, A] — shows no reversal edge on
> six liquid daily tapes (2001/2010→2026): the fade is statistically indistinguishable from a
> coin flip, fails a Bonferroni-corrected comparison against a drift-matched base rate 7/7 times,
> loses money to a single 5 bps cost, and does not beat a placebo control built from the identical
> pivots with an arbitrary, non-0.886 target.

## What we tested

The Bat harmonic: five confirmed swing pivots X, A, B, C where B retraces XA by 0.382-0.50 (a
shallower pullback than Gartley's 0.618), C retraces AB by 0.382-0.886 (the shared XABCD band),
and **D retraces the *original* XA leg by 0.886** — deep, but always staying inside the X-A
range, unlike Butterfly/Crab (which overshoot X). Carney calls this the most reliable pattern in
the harmonic-trading family. On SPY, QQQ, AAPL, MSFT, TSLA and NVDA daily bars (the identical
basket as siblings [698-abcd-harmonic](../698-abcd-harmonic/) and
[699-butterfly-harmonic](../699-butterfly-harmonic/)), we detect every such candidate off a
percentage-threshold zigzag (confirmed pivots only — no look-ahead), scan forward for the first
touch of the projected D, and measure the forward return of fading there — against **two**
independent controls: a random-day base rate matched to the same directional mix (Bonferroni-
corrected across 7 looks), and a placebo arm using the identical pivots with an arbitrary,
off-0.886 D-target.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Bat pattern actually looks like on a chart, why Carney says it's the "safest" harmonic, and what the real tape shows when you actually fade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pivot-confirmation mechanics, the HAC/Welch splits, the Bonferroni correction, the per-instrument breakdown, the cost sweep, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bat_harmonic/`](bat_harmonic/). No survivorship — six currently-listed, individually
named large-cap/ETF tickers, not a membership-conditioned panel.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
