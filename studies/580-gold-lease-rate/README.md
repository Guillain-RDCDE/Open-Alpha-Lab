# Study 580 — Gold-Lease-Rate 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the lease rate lead gold? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | **No real tape exists** — the LBMA discontinued the daily **GOFO** benchmark on 2015-01-30, and no free continuous lease-rate series is reachable — so this is **synthetic-only** and the axis is **capped at WEAK** (`REAL` needs a robust *t* ≥ 2 on a real tape). On the honest **null** world the lead-lag is flat (slope-*t* **+0.08**, corr **+0.005**, R² **0.000**, placebo *p* **0.936**). The engine *is* faithful: plant the folklore and slope-*t* jumps to **+7.73**, with mean *t* over **25 seeds** rising monotonically from **−0.43** (null) to **+11.3**. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no lease-rate ETF or clean product; the implied rate is quoted in arrears with a wide spread; and the operational long-flat rule **underperforms passive gold** on the null world (**+4.7%/yr net** vs **+9.2%/yr** buy-&-hold) — it sits in cash 56% of the time and forgoes drift. Nothing to harvest. |

> **In one sentence:** the gold lease rate — the cost to borrow bullion — is a clean, seductive lead-lag story, but the one public source of it (LBMA GOFO) was switched off in 2015, so the desk can only build the machinery (a lagged predictive regression, a costed long-flat rule, a placebo, a lag sweep, a seed-robust control) and prove it *would* catch a real effect — while honestly stamping `WEAK × MIRAGE` for want of a real tape.

## What we tested

Commodity-microstructure folklore holds that the **gold lease rate** (the cost to borrow physical
metal, classically implied as `LIBOR − GOFO`) is a **leading indicator** for the gold price — a
borrow-cost spike (physical scramble, backwardation) foreshadowing a rally. Because the LBMA
**discontinued the daily GOFO benchmark on 2015-01-30** and no free, continuous lease-rate tape
exists, this study is **synthetic-only**: a deterministic generator (seed 580, 300 monthly periods)
plants (`lead_beta > 0`) or withholds (`lead_beta = 0`) the lead-lag, and the engine runs a
predictive regression of forward gold returns on the *lagged, already-public* lease-rate *z*, a
long-flat trading rule with one-way costs and a buy-&-hold benchmark, a **label-shuffle placebo**
null, a **multi-lag robustness sweep**, and a **seed-robust synthetic positive control** (25 seeds)
that recovers a planted effect and stays flat at the null. *The data limitation is stated openly on
the SIGNAL axis and caps it at `WEAK`. Distinct from the desk's gold **ratios** —
[113 gold-silver](../113-gold-silver-ratio/), [305 gold-oil](../305-gold-oil-ratio/),
[388 lumber-gold](../388-lumber-gold-ratio/) — and the [208 gold-miners](../208-gold-miners/) equity
beta: this is a microstructure **carry/borrow** signal.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the gold lease rate is, why "borrow cost leads price" sounds clever, why the data was switched off in 2015, and what a *null* result looks like |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the lagged predictive regression, the placebo null, the costed long-flat rule vs buy-&-hold, the lag sweep, and the seed-robust synthetic positive control |

The reproducible headline run (synthetic null, fp `756d06b1b283`; positive control, fp
`e3ed82fc53fe`; as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
lives in [`gold_lease_rate/`](gold_lease_rate/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`gold_lease_rate/`](gold_lease_rate/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
