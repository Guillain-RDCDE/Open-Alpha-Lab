# Study 558 — Failures-To-Deliver 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do FTD spikes foreshadow a pop? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the reproducible null world the post-spike **abnormal** forward return is **−0.03%** (5d, market-adjusted), one-sample *t* **−0.14**, placebo *p* **0.513** — dead centre. **No free real FTD tape exists** (SEC bulk CUSIP flat file, no prices), so this is **synthetic-only** and can never clear the `REAL` bar (robust *t* ≥ 2 on a real tape). |
| **Tradability** — does the squeeze trade pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to harvest before costs (abnormal −0.03%); the FTD-spiking name is exactly the expensive-to-borrow tail, so net (5 bps/leg + 300 bps/yr borrow) is **−0.19%**. And the raw signal isn't even reachable — the FTD file is semi-monthly, lagged, CUSIP-keyed, price-free. |
| **"Squeeze predictor?"** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A horizon × threshold **sweep on pure noise** manufactures a spurious \|*t*\| > 2 (the *wrong* sign) on the 1-day cut; across 30 null seeds ~1 in 15 individually prints \|*t*\| > 2 with no effect present. The famous post-spike chart is a multiple-comparisons mirage. |

> **In one sentence:** the meme-stock-era belief that a **failures-to-deliver spike** signals a trapped-short **squeeze** doesn't survive an honest event study — on a faithful synthetic tape the post-spike abnormal return is **−0.03%** (*t* −0.14, placebo *p* 0.51), while cherry-picking one horizon/threshold cut on *pure noise* fakes a \|*t*\| > 2; the engine catches a **planted** squeeze past *t* +2, so the flat null is a true negative, and with no free FTD data the ceiling is `NONE` anyway.

## What we tested

Microstructure folklore (2020–21 meme-stock era): a **spike in failures-to-deliver** (FTD —
shares a seller owed at settlement but didn't deliver) means shorts are cornered, so a **short
squeeze** is imminent and forward returns after an FTD spike should be large and positive. We test
it as a market-adjusted **event study**: flag FTD-spike days from a causal trailing z-score, take
the forward 5-day cumulative return, subtract the cross-name (market) return to isolate the
*abnormal* post-spike pop, enforce a **refractory period** so overlapping event windows can't
inflate the *t*, and run a one-sample *t*, a **label-shuffle placebo** null, a horizon × threshold
**robustness sweep**, costs + a punitive short borrow, and a deterministic, seed-robust synthetic
positive control that plants a pop and proves the engine catches it. **Data caveat, on the Signal
axis:** the SEC fails-to-deliver file is a bulk, semi-monthly, CUSIP-keyed flat file with no price
panel and no retail endpoint, so no free real FTD tape exists — this study is **synthetic-only**
(it can never be `REAL`). *Distinct from [213 Meme-Stocks](../213-meme-stocks/) (the return
phenomenon) and [262 Short-Interest](../262-short-interest/) (short interest / days-to-cover);
this is the **FTD-spike-as-squeeze-predictor** microstructure signal specifically.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a failure-to-deliver is, why "fails ⇒ squeeze" feels true, and why the post-spike pop vanishes when you look honestly |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the market-adjusted event study, the placebo null, the horizon × threshold sweep that fakes significance on noise, costs + borrow, and the seed-robust synthetic positive control |

The reproducible headline run (null world, 60 names × 756 days, panel fp `c0ab49b6db5b`, as-of
2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery lives in
[`failures_to_deliver/`](failures_to_deliver/).

---

*Sources & literature map: [docs/references.md](docs/references.md). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
