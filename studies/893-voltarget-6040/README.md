# Study 893 — Vol-Target 60/40 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the thermostat improve the balanced book? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Point estimates all favour it (excess Sharpe **0.682 → 0.808**, drawdown **−29.6% → −24.2%**) and the **drawdown/risk-control benefit is real & robust** — but the *improvement* is sub-significant: leverage-clean spanning-alpha HAC ***t* = 1.92 (< 2)**, bootstrap Sharpe-diff **CI [−0.086, +0.323] straddles zero**, and the edge **fades across eras** (+0.22 pre-2015 → +0.06 after) and thins as the vol window lengthens. Single-cycle, GFC-anchored, BIL-bounded ~19-yr sample. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The **drawdown cut is bankable** on the two most liquid ETFs alive (turnover ~9×/yr, capacity ~unbounded, −24% at every cost level) — but the thin Sharpe edge is **leverage-financed** (avg 1.36×, levered 74% of days) and a 50 bp borrow + a few bp turnover eats it from **+0.11 to +0.02 by 10 bp**. Real but thin/decaying → Fragile, not Investable. |
| **A free lunch?** | ![Risk-managed](https://img.shields.io/badge/Risk--managed-8b949e?style=flat-square) | No. It re-times risk you already carry; the certain product is a smoother ride (2008 −14.9%→−13.2%, 2022 −17.0%→−14.1%), not a certified Sharpe pickup. |

> **In one sentence:** bolting the equities vol-target ([Study 16](../../16-storm-shy/)) onto the static 60/40 ([Study 97](../../97-balancing-act/)) genuinely **shaves the drawdown** (−29.6% → −24.2%, in every era and every crash), but the *Sharpe* improvement it's sold on is **sub-significant** (leverage-clean *t* = 1.92, bootstrap CI straddles zero, fades post-2015) and leans on leverage a borrow spread erodes — a **Weak** signal in a **Fragile** vehicle you run for the ride, not the alpha.

## What we tested

The classic 60/40's risk is *not* constant — it doubles in a crisis. So we run a **volatility thermostat** on it: scale the whole **60% SPY / 40% IEF** book's exposure by `σ_target / σ̂_{t−1}` (trailing-21-day realized **portfolio** vol, lagged one day, capped 2×) to hold risk near a constant target, and race the re-timed book against the static 60/40 — both **excess-of-cash** (minus BIL), total return, 2007-05-31 → 2026-06-30. The target is set to the static's own realized vol so the drawdown comparison is at **matched risk**. Significance is the leverage-clean **Moreira–Muir spanning alpha** (a plain return-difference *t* is contaminated by the `E[1/σ̂]>1/E[σ̂]` level tilt), a circular block-bootstrap CI on the Sharpe difference, a two-era decay cut, and a cost + borrow sweep. The offline control is a seeded 60/40 world whose *portfolio* vol clusters around a regime-independent drift (flat-vol twin = null). **Dedup:** distinct from [16-storm-shy](../../16-storm-shy/) (same overlay on **equities**, not the balanced book), [97-balancing-act](../../97-balancing-act/) (the **static** 60/40 baseline this tries to beat), [591-vol-managed-portfolio](../../591-vol-managed-portfolio/) (Moreira–Muir `c/RV` on a **single index**, monthly), and [68-all-weather](../../68-all-weather/) (**risk parity** re-weighting *between* assets, not scaling the whole book). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why holding less in storms can smooth the ride, the drawdown that shrinks in every crash, and the honest catch that the Sharpe lift is not certain |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the matched-risk race, the leverage-clean spanning alpha (and why not a return-diff *t*), the bootstrap Sharpe-diff CI, the two-era decay, the window sweep, the costed timer, and the 30-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (SPY/IEF/AGG/BIL, 2007–2026, fingerprint `a874e54fa109`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`fetch()` once to populate the cache); the offline machinery proof runs on the synthetic world in [vt6040/data.py](vt6040/data.py).

---

*Engine: [`quantlab/`](../../quantlab/) + [`vt6040/`](vt6040/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
