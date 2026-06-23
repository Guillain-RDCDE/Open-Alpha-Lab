# Study 387 — Economic-Surprise-Index 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does "data beating" predict higher stocks? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Forward returns after a 'beat' month are positive and the 12-month win-rate is **88%** — but the *excess* over the **82%** base rate is only ~1.4pp, **fails t ≥ 2** (Welch *t* = **1.00**, placebo *p* = **0.13**), is **~zero at 1–3 months** (where the news actually moves), and **inverts** at a loose threshold. A positive-but-insignificant estimate, not an edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A "hold-when-ESI>0" timing rule earns a **lower Sharpe than buy-and-hold** (**0.66 vs 0.78** long/flat; **0.07** long/short) — *before* even crediting the cash yield it ignores while flat. Sitting out an up-drifting market on 'miss' months costs more than the surprise returns. |
| **"Free nowcast"?** | ![Busted](https://img.shields.io/badge/Free_nowcast%3F-Busted-8b949e?style=flat-square) | A surprise measured against a **trailing-average** consensus is mechanically the market's own **post-recession recovery drift** in a costume — priced within minutes of each release, sampled monthly, and indistinguishable from luck. |

> **In one sentence:** "buy stocks when the economic data keeps beating expectations" looks plausible because surprise indices run hot exactly when markets are recovering — but on a transparent CESI proxy built from six FRED series, the forward-return excess over the base rate is small, insignificant (t = 1.0), zero at the horizons the news moves, and a timing rule built on it *loses* to buy-and-hold, so it is real-as-narrative, weak-as-edge, and a mirage to trade.

## What we tested

Citi's **Economic Surprise Index** is proprietary and there is no free history of analyst forecasts, so we **construct a transparent proxy**: from six public monthly FRED real-activity series (payrolls, industrial production, retail sales, consumer sentiment, housing starts, durable orders) we take each month's change **minus its trailing-12-month average** — a clearly-labelled stand-in for the Street's consensus — then z-score and average them into one **Economic Surprise Index (ESI)**. A 'beat' month is ESI > 0 (data above its own recent pace). Over **33.3 years** (1993–2026, **401** months) we measure forward 1/3/6/12-month SPY returns after a beat vs the unconditional base rate, with a 1-month entry lag, a Welch *t*, and a 20,000-draw **placebo** null; we also race an ESI-timing rule against buy-and-hold net of costs. A deterministic synthetic control with a *planted* edge confirms the engine is faithful **and** that a noise signal stays flat (t = 0.47 with no edge; t = 2.22 when an edge is planted).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a surprise index is, why "beat = buy" feels right, and why a slow trailing-average gauge is really just the recovery rally in disguise — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ESI construction, conditional vs unconditional forward returns, a Welch *t* + placebo null, an ESI-timing-vs-buy-and-hold Sharpe race net of costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`economic_surprise_index/`](economic_surprise_index/). The surprise index here is an explicit **proxy** (six FRED series vs a trailing-average consensus), not Citi's CESI. SPY is price-only. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
