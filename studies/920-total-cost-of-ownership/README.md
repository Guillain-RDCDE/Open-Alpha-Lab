# Study 920 — Total Cost of Ownership 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the cheap wrapper's advantage on the tape? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | On the common window the cheap wrapper's realised tracking difference is **+5.78 / +6.60 / +7.19 bp/yr** (IVV, VOO, QQQM over their pricier twins) at *t* = **+2.92 / +3.40 / +5.27**, **positive in 5/5 years on each pair** (sign-test *p* = 0.031), Student-*t* intervals clear of zero, within ~2 bp of the published fee gaps — while the same-fee **placebo (VOO vs IVV) prints +0.82 bp/yr (*t* = +0.75, 3/5 years)** and is still negative after three years in the overlapping race. Caveats, all load-bearing: the expensive leg is a **unit investment trust in every pair that shows a gap**, so this is fee gap **+ trust cash drag, unseparated**; the result is **estimator-conditional** (the raw cumulative drift gives IVV/SPY +3.38 vs the placebo's +4.69 — only the ex-ante complete-year rule, justified by a **+23.6 bp six-month stub artefact on a zero-gap pair**, separates them); the **full pre-2020 histories fail |*t*| = 2**; all four pairs share the *same* five years; the funds are survivors. |
| **Tradability** — is it bankable? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | Not because the tape pins the number down — it does not — but because the act is a **purchase decision with no forecast, no timing, no turnover and no capacity limit**, on a fee differential that is contractual and merely *confirmed* by the tape. Break-even is **35–44 trading days** at a 1 bp round-trip differential (42–63 days on overlapping windows with one execution lag), under nine months at a punitive 5 bp. What the tape cannot promise is the size: at the honest interval's pessimistic end the same 1 bp takes **74 / 208 / 898 days** to repay. And it is **not a trade** — the long/short version dies above ~5 bp/yr of borrow, where a **same-fee placebo pair "harvests" +3.58 bp/yr, more than two of the three real pairs**. |

> **In one sentence:** the cheap clone really does deliver 6 to 7 basis points a year more than its pricier twin — the size the prospectuses imply, absent from the placebo pair that charges the same fee — so anyone holding past a quarter should own the cheap wrapper; but the tape that measures it is dirtier than the thing measured, the advantage cannot be separated from the 1993 trust's cash drag, and the whole prize is a basis point a month that vanishes the moment you try to trade it rather than simply own it.

## What we tested

Total cost of ownership = **expense ratio** + **realised tracking difference** + **round-trip
spread**. Only the middle term is on a daily-close tape, so we measure it — cheap minus liquid,
in bp/yr, on total-return closes — for **IVV/SPY**, **VOO/SPY**, **QQQM/QQQ** and the same-fee
**VOO/IVV placebo**, then convert it into a **break-even holding period** against a *swept*
round-trip spread assumption (0–10 bp) and check that curve against every overlapping holding
window on the tape with one execution lag. Fees are prospectus numbers, spreads and borrow are
quote-level numbers: all three are labelled ASSUMPTIONS and swept. Three chained estimators
(cumulative, annual, monthly HAC) that **disagree by 2–3× on the real tape**, a stub
decomposition that shows why, a Student-*t* interval beside the block bootstrap because five
annual observations break the bootstrap, median and trimmed-mean sensitivities, an era cut, a
borrow sweep and a calibrated synthetic control. **SPLG is excluded** — Yahoo's history for it
was reset to a single session at build time. **Dedup:**
distinct from **913-tracking-difference-persistence** (same wrappers, but a *ranking-persistence*
question — we measure the level, not who wins next year), **378-etf-nav-premium** (transient
premium to a NAV proxy, which is our noise floor rather than our signal), **379-etf-lead-lag**
(a timing signal), **621-share-class-spreads** (two share classes of one company, held by a
conversion bound) and **622/624** (what *exotic* wrappers cost). Common window
2020-10-13 → 2026-06-30, as-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the sticker fee is not the price, the two-month break-even in plain language, the placebo that proves the ruler works, and where the saving stops being worth chasing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the chained-period estimator and the stub decomposition that decides between its three variants, the bootstrap-vs-Student-*t* interval fight, the annual/median/trimmed sensitivities, the adjustment-artefact autopsy, the break-even curve at both ends, the overlapping race with one lag and where its HAC *t* stops meaning anything, the borrow sweep with its placebo, and the live calibration check |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`tco/`](tco/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
