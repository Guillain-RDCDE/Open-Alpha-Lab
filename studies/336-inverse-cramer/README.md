# Study 336 — Inverse-Cramer 📺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | On the meme's own one-month horizon the curated fade is **+420 bps/call** but HAC *t* = **+1.47** — below the bar — with a bootstrap CI **straddling zero** ([−114, +937]). It clears *t* = 2 only at a snooped longer horizon, on 20 hand-picked calls. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Unrunnable as stated: the call list exists only in **hindsight**. Costs barely register (+400 bps at 10 bps), but you can't trade a meme's "his worst calls" table — a live, *all-calls* fade is the real test, and this isn't it. |
| **Selection bias?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A synthetic **coin-flip** pundit, curated to his worst calls, prints a fade at HAC *t* = **+4** with zero predictive content. The headline edge is consistent with pure curation. |

> **In one sentence:** fading the loudest pundit *looks* like an edge on a hand-picked list of his most famous misses — but on its own one-month horizon it doesn't clear the significance bar, it's untradable because the list only exists in hindsight, and a coin-flip pundit produces the same "edge" once you publish only his worst calls.

## What we tested

The folk thesis — popular enough to spawn a real (now-closed) ETF, **SJIM**, the *Inverse Cramer Tracker* — that CNBC's Jim Cramer is so reliably wrong that doing the **opposite** of his on-air calls is an edge. We take it literally on a **hardcoded, curated table of 22 notable calls** (Bear Stearns "it's fine!", Coinbase and Meta near the top, …): the "fade" trades opposite each stated direction, entered one session after the call (one shift) and held a fixed forward window, pinned against a **random-direction control** on the same calls, swept for cost, and checked against a deterministic synthetic universe. The table is curated *for memorability* — selection on the outcome — and that bias is the study's central confound; the synthetic control quantifies exactly how much "edge" pure curation manufactures.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the meme, why the curated list looks unbeatable, and the coin-flip pundit that fakes the same edge |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* + bootstrap CI, the random control, horizon sensitivity, the selection-bias synthetic, capacity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`inverse_cramer/`](inverse_cramer/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
