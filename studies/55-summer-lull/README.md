# Study 55 — Summer-Lull ☀️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do equities really do better in winter? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Real and persistent, but soft. Winter (Nov–Apr) earned **86 bp/mo (+5.3%/yr)** vs summer's **45 bp/mo (+2.7%/yr)** over 98 years — a durable gap, but only **Welch t 1.3**. |
| **Tradability** — does "sell in May" beat buy-and-hold? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. It matches buy-and-hold's Sharpe (0.43 vs 0.42) but compounds to **a third less** (4.5%/yr vs 6.3%) — sitting in cash through a season that still pays +2.7%/yr. |
| **"Sell in May is good advice"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Summer returns are **positive**. Skipping a positive season can't improve risk-adjusted performance — the lower drawdown is just less exposure. |

> **In one sentence:** the Halloween seasonal is one of the most durable patterns on the bench — winter has beaten summer for nearly a century — but "sell in May and go away" is still bad advice, because summer pays a positive +2.7%/yr: sitting it out just lowers your wealth at the same Sharpe.

## What we tested

**"Sell in May and go away"** (the Halloween indicator; Bouman & Jacobsen 2002): equities earn more Nov–Apr than May–Oct, so you should hold stocks in winter and step aside in summer. We test it on **98 years** of S&P 500 data (1928–2026): the winter-vs-summer gap with a Welch t, and the tradable rule (hold Nov–Apr, cash May–Oct) against buy-and-hold — paying special attention to the fact that summer returns are *positive*, not negative. We split pre/post-2000 for decay. The offline control is a synthetic world with a known winter premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a real seasonal can still make "sell in May" a money-loser |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the winter-summer split with its Welch t, the sell-in-May-vs-buy-and-hold race, the persistence, the exposure illusion |

The fingerprinted real-data run (S&P 500 1928–2026, fp `b5bbd3b8ce29`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [summer_lull/data.py](summer_lull/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
