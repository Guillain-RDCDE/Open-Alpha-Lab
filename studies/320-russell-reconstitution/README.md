# Study 320 — Russell-Reconstitution

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | IWM 5-session run-up excess **+36 bps**, HAC *t* = **+0.89**; reconstitution day & give-back ≈ 0; same-month June control unchanged; sign flips across sub-periods. Indistinguishable from noise at n = 26. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even at face value the front-run rule earns ~**+0.5%/yr** while sitting in cash 360 days — and there is no significant edge underneath it. The forced flow is real; the harvestable *price* move is not. |
| **Front-runnable?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The famous one-directional reconstitution flow is huge in *shares* but the index ETF is already efficiently priced by the time you could trade it. Big volume, no drift. |

> **In one sentence:** the late-June Russell reshuffle is the most predictable forced flow of the year — and on the small-cap *index* there is simply no front-runnable price move to capture once you test it honestly against the right null.

## What we tested

Every June, FTSE Russell rebuilds its US indices and the new membership goes effective after the close of the late-June *reconstitution Friday* — forcing every tracking fund to trade to the new weights in one of the year's highest-volume closing auctions. The date is published years ahead and the flow is one-directional, so the folklore (and a long line of sell-side "Russell rebalance trade" notes) says you can **front-run it**: buy the small-cap index in the run-up and sell on the event. We take that literally on IWM (iShares Russell 2000) daily bars, 2000–2025 — 26 reconstitutions — running the run-up, reconstitution-day and give-back windows against the unconditional rolling-window baseline *and* a same-month June control, with a one-sample t-test, a Newey-West HAC t on the excess, a permutation null, sub-periods, and a synthetic positive control. **Distinct from [Study 249](../../249-index-inclusion/)**, which tests single-name S&P 500 *additions*; this is a calendar event on the *whole index ETF*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the reshuffle, why "huge forced flow" feels tradable, and why the index doesn't actually move |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-vs-baseline HAC *t*, same-month control, window-length & sub-period sweeps, the n = 26 power problem, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`russell_reconstitution/`](russell_reconstitution/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
