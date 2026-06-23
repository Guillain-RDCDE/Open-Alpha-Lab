# Study 363 — PEAD-Drift 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the drift exist? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Sorted on the **fundamental EPS surprise**, the top-minus-bottom drift is **+1.34% at 20 days** (one-sample *t* = **2.96**, label-shuffle *p* ≈ **0**) and **+1.71% at 60 days** (*t* = **2.10**). Clears the **t ≥ 2** bar, robust to quintile count and to a within-quarter block placebo (*p* = **0.016**). PEAD is genuinely there — with an explicit **survivorship** caveat (large-cap survivors tilt the long leg). |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Net of one-way costs × per-event turnover (+ borrow), the **short horizons flip negative** (−0.24% at 1 day) and only **20–60-day holds survive** (**+0.90% / +1.19%** per event). A thin net spread on a **30-name** survivor basket, earned only over multi-week holds, leaning on the long leg — real but operationally delicate. |
| **"Always drifts the same way"?** | ![Busted](https://img.shields.io/badge/Always_drifts%3F-Busted-8b949e?style=flat-square) | Sort on the **price gap** you can *see* (the folk recipe "buy the pop and ride it") and it **never clears t = 2 and goes negative at 60 days**. The drift lives in the *fundamental* surprise and only after ~20 days — the anomaly is right, the folklore's *mechanism* is wrong. |

> **In one sentence:** post-earnings drift is one of the rare folk effects that is genuinely real — sorted on the fundamental EPS surprise, the top-minus-bottom 20-day drift is +1.34% at *t* = 2.96 and survives a quarter-block placebo — but it is thin, horizon-gated (nothing in the first week), barely survives costs on a tradable large-cap basket (and only at 20–60 days), and the naive "buy the visible gap and ride it" version doesn't work at all.

## What we tested

We rebuild PEAD as a clean event study on a fixed **30-name large-cap basket**: per name we pull every quarterly earnings date with its reported **EPS surprise** (and, separately, the one-day post-announcement **price gap**), sort events into surprise quintiles, and measure the forward **1 / 5 / 20 / 60-day** drift of a top-minus-bottom long-short — entering one day after the reaction is public (no look-ahead). The Signal axis tests the long-short against zero with a one-sample *t*, a 20,000-draw surprise-label placebo, and a clustering-aware within-quarter block placebo; Tradability charges one-way costs × per-event turnover plus short-leg borrow. A deterministic synthetic control with a *planted* drift confirms the engine is faithful and that zero edge cannot fake significance. Survivorship (the basket is names still trading in 2026) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an earnings surprise is, what "drift" means, why the *fundamental* beat drifts but the *visible pop* doesn't, and why costs eat the fast money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile long-short on the EPS surprise, forward 1/5/20/60-day drift, a one-sample *t* + label-shuffle & block placebo nulls, costs × turnover, the gap-sort myth check, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pead_drift/`](pead_drift/). Surprise proxies are the reported EPS `Surprise(%)` (headline) and the one-day post-announcement gap (myth-check). Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
