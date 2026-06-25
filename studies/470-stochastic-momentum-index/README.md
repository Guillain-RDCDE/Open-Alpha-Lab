# Study 470 — Stochastic Momentum Index 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the SMI-turn beat the drift? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | The "rising out of oversold" turn **robustly beats a drift-matched random baseline** at the desk's *t* ≥ 2 bar on **10 and 20 days** — and *seed-robustly*: averaged over **20 baseline seeds** the turn-vs-random Welch *t* = **+2.84 / +2.71** with **every** seed clearing 2 (min +2.33 / +2.10), not a single-seed fluke (Δ = **+77 / +102 bps**, positive in **all five** names). The **structure placebo rejects hard** (count-matched random-date entries beat the real turn in **0/300** draws, *p* = 0.0033). It is a genuine short-horizon bounce, not pure beta. (5d is borderline — seed-avg *t* = +2.22, only 70% of seeds ≥ 2; 60d not significant, *t* = +1.38.) |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Real but **thin and generic**: only **226 turns in 21 years** (~2/ticker/yr), Δ thin in 2 of 5 names, it fades by 60d, and — decisively — the parameter placebo shows it is **not** Blau's specific SMI but the oversold-dip *family* doing the work (*p* = 0.42). Too sparse and parameter-agnostic to deploy as a standalone rule. |
| **"Does the SMI forecast turns?"** | ![Mixed](https://img.shields.io/badge/Forecasts_turns%3F-Mixed-dab617?style=flat-square) | Half right. A real oversold bounce exists (signal-vs-random passes) — but scrambling the SMI's tuning leaves it intact (**p = 0.42** of random-tuned cousins match or beat it), so the *family* forecasts, not Blau's specific construction. |

> **In one sentence:** Encode William Blau's Stochastic Momentum Index mechanically and fire the "buy the rising-out-of-oversold turn" rule 226 times across 5 ETFs over 21 years, and — unusually for this desk — it **robustly beats buying on random days** by ~75–100 bps at 10–20 days (seed-averaged Welch *t* = +2.84 / +2.71, every one of 20 baseline seeds clearing 2; structure placebo *p* = 0.0033): a *real* short-horizon oversold bounce, not a single-seed fluke. But scramble the SMI's *tuning* and any cousin oscillator does just as well (*p* = 0.42), so the credit belongs to the broad oversold-reversion effect, not to Blau's specific SMI — and with only ~2 signals/year/ticker it is far too thin to trade.

## What we tested

We encode the tightest mechanical version a proponent would accept. The SMI is Blau's double-smoothed range oscillator — distance of the close from the **midpoint** of the N-day high/low band, two EMAs deep, scaled to ±100 (N = 13, s1 = 25, s2 = 2, oversold gate −40); it is **causal** (past bars only). A long fires when the SMI was below −40 on the prior bar and **turns up**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **turn vs a drift-matched random-entry baseline**, reported as the **seed-averaged Welch *t* over 20 baseline seeds** (a single seed can throw a lucky *t* > 2 that isn't a real edge — cf. Study 452 — so the legitimate test is seed-robustness; here every seed clears 2 at 10/20d), plus a **structure placebo** (count-matched random-date entries, *p* = 0.0033) and a **scrambled-parameter placebo** that randomises the SMI's look-back/smoothing while keeping the same oscillator family. Tradability charges costs on every turn. A deterministic synthetic control with a *planted* oversold bounce (keyed off the same causal SMI) proves the detector is live (edge 0 → *t* ≈ 0.75; planted bounce → *t* = +2.56), so the real-tape signal is a genuine detection.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the SMI is, why most chart tools are just drift, the turn-vs-random race the SMI *passes*, and the tuning-scramble that shows it isn't the specific indicator — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal SMI, one-sample HAC *t* vs the beta trap, the (passing) random-entry Welch test, the scrambled-parameter placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`stochastic_momentum_index/`](stochastic_momentum_index/). The SMI is causal (no future bars); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument timing study, so the large random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
