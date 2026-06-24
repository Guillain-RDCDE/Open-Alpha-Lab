# Study 445 — Elliott Wave 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the five waves predict anything? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The tightest mechanical rule — buy **wave 3** after a Fibonacci-validated impulse 1-2, marked by a 5% ZigZag — produces **no** detectable edge on **661** entries pooled across 8 indices/33 years. The largest statistic anywhere is **HAC *t* = 1.63 at 40 days** (and that's just market beta); shorter horizons are noise or negative, and a same-bars coin matches the wave count **30–84%** of the time. Robust to the Fibonacci band and ZigZag threshold. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There's no edge to charge costs against. The one mildly-positive number (40-day) is indistinguishable from buy-and-hold, and costs flip the fast versions negative. Nothing to trade. |
| **"Five waves predict the next move"?** | ![Not_Supported](https://img.shields.io/badge/Five_waves_predict%3F-Not_Supported-8b949e?style=flat-square) | Neither forward-looking claim survives — wave-3 extension after a 1-2, *or* a correction after a completed five-wave impulse — on 33 years of daily data. A synthetic control proves the harness **would** catch a planted wave-3 (*t* = 7.02), so this is a true negative, not a broken test. |

> **In one sentence:** Elliott Wave is *irreducibly subjective* — the count is only knowable in hindsight — so we tested the tightest mechanical version its proponents accept (ZigZag pivots + a Fibonacci wave-2 filter + the "buy wave 3" entry), and across 661 entries on 8 broad indices over 33 years it predicts **nothing** out-of-sample (max HAC *t* = 1.63, beaten by a coin at the same swing points), even though the same engine lights up at *t* = 7.02 when we *plant* a wave-3 edge.

## What we tested

Elliott Wave Theory says price unfolds in fractal **five-wave impulses** (1-2-3-4-5) followed by **three-wave corrections** (A-B-C), with **Fibonacci** ratios governing the legs (wave 2 retraces ~50–61.8% of wave 1; wave 3 is the strongest leg). The theory is irreducibly subjective — two analysts label the same chart differently and the "right" count is only clear after the fact — so there is no single falsifiable rule. We encode the tightest mechanical version proponents do accept: a standard **5% ZigZag** swing detector marks the pivots, a Fibonacci filter validates the wave-2 retracement (38.2–78.6%, with the textbook 50–61.8% inside), and we test the one tradable claim — after a clean impulse 1-2, **wave 3 extends in the impulse direction** (plus the "sell the completed five-wave" correction claim). Entry is one day after the pivot is confirmed (no look-ahead); we report one-sample and HAC *t*, a **same-bars random-direction placebo** (does the count beat a coin at the identical swings?), a Fibonacci-band and ZigZag-threshold robustness sweep, and costs. A deterministic synthetic control with a *planted* wave-3 confirms the engine can detect the structure when it's there.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Elliott waves are, why the count is a hindsight game, what a ZigZag actually marks, and why "buy wave 3" does no better than a coin — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ZigZag + Fibonacci wave-3 entries, forward 5/10/20/40-day returns, one-sample & HAC *t* + a same-bars coin placebo, Fibonacci-band & threshold robustness, the completed-impulse correction test, costs, and a planted-edge faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`elliott_wave/`](elliott_wave/). Mechanical proxy = 5% ZigZag pivots + Fibonacci wave-2 filter; the method itself is irreducibly subjective and we say so. Tapes are auto-adjusted (total-return) daily closes; price indices, no cross-sectional survivorship sort. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
