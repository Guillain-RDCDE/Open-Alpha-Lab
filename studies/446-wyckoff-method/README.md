# Study 446 — Wyckoff Method 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the Spring/Upthrust events predict the move? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The tightest mechanical rule — a ZigZag-bracketed **trading range** plus the **volume-confirmed Spring (buy) and Upthrust (sell)** — produces **no** detectable edge across **404** events on 8 index/ETF tapes over up to 33 years. The best statistic anywhere is **HAC *t* = 0.89** (20-day), and a coin at the same bars does *better*; the placebo *p* never drops below **0.13**. The one positive leg (the Spring, *t* = 2.29) **fails the HAC bar (1.77)** and is just the bull-market long bias — the short Upthrust leg loses in the same tapes. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There's no edge to charge costs against. The events are rare (so costs are trivial) but the net combined signal is *t* ≈ 1.0 — nothing to trade. |
| **"Are accumulation/distribution phases detectable?"** | ![Not_Supported](https://img.shields.io/badge/Phases_detectable%3F-Not_Supported-8b949e?style=flat-square) | The two canonical, volume-confirmed events fire plenty but neither leg foretells the move Wyckoff claims — and the volume "tell" the method rests on **adds nothing** (removing it *raises* the *t*). A synthetic control proves the harness **would** catch a planted markup (*t* = 5.20), so this is a true negative, not a broken test. |

> **In one sentence:** the Wyckoff method is *irreducibly subjective* — the phase labelling is only clear in hindsight — so we tested the tightest mechanical version its proponents accept (a ZigZag trading range plus the volume-confirmed Spring and Upthrust), and across 404 events on 8 tapes over up to 33 years it predicts **nothing** (best HAC *t* = 0.89, beaten by a coin, placebo *p* ≥ 0.13), with the only positive leg explained by bull-market drift — even though the same engine lights up at *t* = 5.20 when we *plant* a markup edge.

## What we tested

Wyckoff's method describes a market cycle of **accumulation → markup → distribution → markdown**, run by a "Composite Operator" who builds positions inside sideways **trading ranges**. The two events every Wyckoff trader watches for are the **Spring** (price briefly breaks *below* a range's support, fails on low volume, and snaps back — a buy that precedes markup) and the **Upthrust** (the mirror image above resistance — a sell that precedes markdown). The full method — phase labelling A–E, the nine buying/selling tests, "effort vs result" — is irreducibly subjective: two analysts annotate the same chart differently and the right phase is only clear after the fact. So we encode the tightest mechanical version proponents accept: a **5% ZigZag** marks pivots, four pivots inside a tight band define a range with a support and resistance, and a **volume-confirmed** Spring/Upthrust (test bar on below-average volume — the "no supply / no demand" tell) fires the trade. Entry is one day after the event (no look-ahead); we report one-sample and HAC *t*, a **same-bars coin placebo** (does the event *label* beat a coin?), a volume-filter and ZigZag-threshold robustness sweep, a per-leg split, and costs. A deterministic synthetic control with a *planted* markup confirms the engine can detect the structure when it's there.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what accumulation/distribution and a Spring/Upthrust are, why the phase count is a hindsight game, and why "buy the Spring" does no better than a coin (and only the long leg even looks alive) — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | trading-range + volume-confirmed Spring/Upthrust events, forward 5/10/20/40-day returns, one-sample & HAC *t* + a same-bars coin placebo, volume-filter & threshold robustness, the per-leg split, costs, and a planted-markup faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wyckoff_method/`](wyckoff_method/). Mechanical proxy = 5% ZigZag trading range + volume-confirmed Spring/Upthrust; the method itself is irreducibly subjective and we say so. Tapes are auto-adjusted (total-return) daily OHLCV; index/ETF tapes, no cross-sectional survivorship sort. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
