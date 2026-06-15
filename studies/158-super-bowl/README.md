# Study 158 — Super-Bowl

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hit-rate **54.2%**, binom p = **0.41**, perm p = **0.35**, Welch t = **+0.63** vs the correct baseline (73% unconditional up-rate); n = 59 too small to detect anything real. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No tradable implementation exists beyond "go long or flat the S&P once a year" — dominated by passive buy-and-hold. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The famous streak (23-for-23 through 1997) was a sampling coincidence: the NFC dominated early Super Bowls during a secular bull market. |

> **In one sentence:** the Super Bowl Indicator is a coincidence dressed as an omen — tested against the honest baseline (the S&P goes up 73% of years anyway), the NFC win signal is statistically indistinguishable from noise at any meaningful sample size.

## What we tested

The Super Bowl Indicator (Krueger & Kennedy 1990): if an NFC (or original-NFL) team wins
the Super Bowl, the S&P 500 supposedly rises that year; if an AFC (AFL) team wins, it
falls. We hardcode all 59 Super Bowl results (1967–2025) in `data.py`, join them with
Shiller S&P 500 annual returns, and test the NFC-win hit-rate and original-NFL hit-rate
against the **correct baseline** — the 72.9% unconditional up-rate, not a 50% coin. We
report a binomial test, a permutation test with 10,000 shuffles, a Welch t-test on annual
returns, and a Bonferroni correction for two simultaneously tested variants. The synthetic
positive control confirms the machinery can find a signal when one is planted; the real
tape confirms there is none here.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the base-rate trap, the streak and its collapse, the correct test in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | binomial test, permutation distribution, Welch t, multiple comparisons, the n=59 power calculation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`super_bowl/`](super_bowl/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
