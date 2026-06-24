# Study 419 — Chaikin Money Flow

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | CMF(20) long/flat on SPY nets Sharpe **−0.01**, HAC *t* = **−0.06**; block-permutation placebo *p* = **1.00**. Indistinguishable from noise. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Loses to a volume-blind SMA 50/200 filter (**+0.68**), MACD (**+0.43**) and buy-and-hold (**+0.51**); no positive break-even cost. |
| **Does flow lead price?** | ![Not supported](https://img.shields.io/badge/Leads_price%3F-Not_supported-8b949e?style=flat-square) | Δ vs the SMA filter = **−0.70** Sharpe; the timing-scramble matches the real signal. CMF coincides with price, it does not lead it. |

> **In one sentence:** Chaikin Money Flow reads where the close sits inside *today's* bar — a contemporaneous fact — and on broad US equities that fact carries no information about *tomorrow's* return: a plain volume-blind moving-average filter, and even doing nothing, beats it.

## What we tested

Marc Chaikin's CMF sums the volume-weighted intrabar close location over a trailing window (default 20) into a [−1, +1] oscillator; the folk rule is **CMF > 0 = accumulation = buy, CMF < 0 = distribution = step aside**, on the premise that *money flow leads price*. We turn that into a long/flat timing rule on SPY (CMF > 0 → long, else cash), decide at the close and earn the next day's return (one execution lag), and race it **net of 2 bps one-way × NAV costs, excess-of-cash** against the obvious simpler benchmarks — an SMA(50/200) trend filter, a MACD filter, and buy-and-hold — over 2000–2026, with a block-permutation placebo on the timing, a cost sweep, a five-name panel, and a synthetic positive control that plants a real flow-leads-price edge to prove the harness can see one.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what CMF measures, the four-way race, why "money flow leads price" doesn't survive the tape |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on each book, the permutation placebo, cost sweep, the panel, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`chaikin_money_flow/`](chaikin_money_flow/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
