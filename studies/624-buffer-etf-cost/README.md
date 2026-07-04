# Study 624 — Buffer-ETF-Cost 🛡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the cost of comfort statistically real? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Every buffer fund trails SPY total return by **5.3–8.3 pp/yr** at HAC *t* = **2.10–3.04** (4-vintage cohort **+7.24 pp/yr**, *t* = **2.44**, lag-robust); the cap visibly bound in **16/21** up periods (mean give-up ~10 pp/period, capped years plateau at ~12% vs SPY up-years ~22%). Panel = the category's surviving flagships (**survivor slice**, named — it flatters the funds). |
| **Tradability** — could you build the insurance cheaper? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The 5–8 pp/yr "cost" is **beta in disguise**: vs a beta-matched SPY/BIL mix the gap collapses to **≈ 0 pp/yr** (all five funds \|*t*\| < 1; Sharpe 0.79 vs 0.82; near-identical vol and drawdown). Fee + forgone dividends (~1.8 pp/yr) were charged — and the option payoff **earned them back** (residual −1.7 pp/yr, mostly the buffered 2022). No deployable "replace it with the dumb mix" edge. |
| **Did any buffer cohort beat the dumb mix it replaces?** | ![Busted](https://img.shields.io/badge/Beat_the_dumb_mix%3F-Busted-8b949e?style=flat-square) | **0/5** funds beat their beta-matched mix significantly (best +0.94 pp/yr at *t* = 0.92; 2/5 positive point estimates, 3/5 negative). A statistical tie — comfort was **fairly priced ex post**, neither free lunch nor rip-off. And the buffer itself was delivered **4/4** times it was called (PJAN 2022: SPY price −19.48% → fund −5.29% vs terms floor −5.27%). |

> **In one sentence:** buffer ETFs kept every promise on the tape — full 15% buffer delivered in all four down years, hard ~12% cap enforced in 16 of 21 up years, and a genuine 5–8 pp/yr shortfall vs the market (HAC *t* ≥ 2.1) — but that shortfall is the beta you chose, not a fee you can dodge: a dumb beta-matched SPY/BIL mix only **ties** them (gap ≈ 0, |*t*| < 1), so the comfort turns out fairly priced and the "build it cheaper yourself" critique fails ex post.

## What we tested

The defined-outcome pitch — and its standard critique. We take **BUFR** (the flagship laddered buffer) plus the four oldest Innovator **Power Buffer** vintages (PJAN/PAPR/PJUL/POCT: 15% buffer on SPY's *price* return, annual reset, 0.79% ER) and run three measurements on the 2018–2026 tape: **(1) mechanical delivery** — fund total return vs SPY price return per completed outcome period, checking the promised floor and cap against the stated terms (25 periods); **(2) the cost of comfort** — Newey-West *t* on the monthly shortfall vs SPY total return, per fund and pooled; **(3) the fair race** — each fund vs the beta-matched w·SPY+(1−w)·BIL mix it supposedly overcharges you for, monthly rebalanced with explicit costs, with a fixed-w grid, lag grid and cost sweep as robustness. Deterministic synthetic controls plant a tunable structuring drag and a known cap/buffer to certify both detectors. TR-vs-TR and price-vs-TR are labeled everywhere; the panel's survivor tilt is named on the Signal axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a buffer ETF actually promises, the payoff scatter that shows every promise kept, what the comfort really cost — and why the "just build it yourself" advice earned exactly nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | outcome-period delivery table, HAC gap tests vs SPY and vs beta-matched mixes, cost decomposition (fee + dividends forgone + option residual), w-grid / lag / cost robustness, synthetic drag & delivery controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`buffer_etf_cost/`](buffer_etf_cost/). Sibling structure studies: [337-covered-call-etf](../337-covered-call-etf/) (income wrapper — sell the upside for a distribution; this study is the defined-outcome wrapper — buy a buffer, accept a cap) and [99-safety-net](../99-safety-net/) (DIY stop-loss insurance). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
