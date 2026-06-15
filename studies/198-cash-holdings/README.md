# Study 198 — Cash-Holdings

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Hedge +12.4%/yr, HAC *t* = +4.57 — nominally passes the bar, but *entirely* attributable to (1) survivorship bias: failed cash-rich firms are absent from the S&P 500 survivor panel, and (2) latent tech/growth factor: the high-cash quintile on the S&P 500 is dominated by Apple, Alphabet, and Microsoft, whose outperformance is unrelated to their cash ratios. The Palazzo (2012) mechanism requires financial constraints that are absent for large-cap S&P 500 names. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Cannot short the disappearing firms that survivorship bias removes; the apparent alpha is a tech-sector overweight more cheaply accessed via QQQ; the alpha estimate is unreliable on this universe. |
| **Survivorship bias?** | ![Upper_bound_only](https://img.shields.io/badge/Survivorship-Upper_bound_only-8b949e?style=flat-square) | Panel is current S&P 500 members projected back. Failed cash-rich firms (high-cash → distress → delisted) are absent. Every headline number is an upper bound. |

> **In one sentence:** the cash-holdings premium nominally appears on the S&P 500 survivor panel, but it is a mirage driven by survivorship bias and a latent tech-sector factor — the financial-constraint mechanism Palazzo (2012) proposes does not operate for large-cap names with open capital-market access.

## What we tested

Palazzo (2012) argues that firms hoard cash because they face high external financing costs, and investors demand a premium for holding these financially constrained firms. We compute Cash-to-Assets = CashAndCashEquivalentsAtCarryingValue / Total Assets from the shared EDGAR cache, sort the current S&P 500 survivors into quintiles on this signal, lag fundamentals by one full year (fiscal year y → calendar year y+1 returns), and test whether the high-cash quintile outperforms the low-cash quintile vs an equal-weight market and a random-portfolio control. A deterministic synthetic tape with tunable cash premium serves as the positive control.

The panel is **survivorship-biased**: it covers only firms that remain in the S&P 500 as of 2026. The high-cash quintile is dominated by large-cap technology firms (Apple, Alphabet, Microsoft) — firms that are cash-rich because they are enormously profitable, not because they are financially constrained. Every headline number is an upper bound.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the cash-premium story in plain English, the survivorship-bias autopsy, the latent tech-factor problem, why large caps are the wrong testing ground |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile return monotonicity, HAC t-stats, random-portfolio null distribution, synthetic positive control sweep, three fatal failure modes |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cash_holdings/`](cash_holdings/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
