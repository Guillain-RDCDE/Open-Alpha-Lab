# Study 217 — Lipstick Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Welch t = **-0.357** (p = 0.72), perm p = **0.67**; the basket *underperforms* SPY in recession months by -0.16%/month — the direction is reversed vs the claim. n = 31 recession months is far too small to detect anything real. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No usable signal; NBER recession calls arrive with a 6-18 month lag; even on-time the strategy hurt, not helped. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Leonard Lauder's 2001 anecdote launched a meme. Cosmetics stocks behave like any other consumer-facing sector in downturns: they fall with the market. |

> **In one sentence:** the Lipstick Index is an anecdote masquerading as a signal — in recession months the cosmetics basket actually underperforms SPY, delivering the opposite of what the folklore claims, with a Welch t of -0.36 on only 31 recession months.

## What we tested

Does lipstick (or nail-polish) spending warn of a coming recession? The Leonard
Lauder Lipstick Index (~2001): cosmetics companies supposedly outperform the broad
market in downturns because consumers "trade down" to affordable luxuries. We proxy
via an equal-weight basket of US-listed cosmetics names (EL, ULTA, COTY, ELF) and
SPY as benchmark, covering 359 months from 1996 to 2025. We compute the monthly
*relative return* (basket minus SPY) and test whether recession months (NBER
definition, hardcoded) show a higher relative return than expansion months. We use
a Welch t-test, a permutation test with 10,000 shuffles, and a synthetic positive
control to verify the machinery can detect an effect when one is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the anecdote, the trade-down hypothesis, why the basket actually falls in recessions, the honest baseline |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Welch t on monthly relative returns, permutation distribution, n=31 power calculation, basket composition bias |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lipstick_index/`](lipstick_index/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
