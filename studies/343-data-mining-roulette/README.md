# Study 343 — Data-Mining-Roulette 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is any mined rule statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No single rule from a 1,000-rule search is certifiable: on a tape with **provably nothing to find**, the luckiest rule still posts Sharpe 0.66 / HAC *t* = 2.20, and the naive count of "winners" (25 on noise, 765 on real SPY) is an artefact of the search, not evidence. |
| **Tradability** — is there an edge the search created? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "best" rule is a snooped pick of a low-capacity micro-effect; it erodes with costs (Sharpe 0.75 → 0.35 at 20 bps) and decays out of sample. The roulette manufactures no harvestable edge. |
| **Does luck mimic skill?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Bonferroni passes **0** rules on the null and White's Reality Check returns *p* = 0.30 — only correcting for the search reveals the champion is exactly what luck produces. The machinery rediscovers a *planted* edge perfectly, so the null result is real, not a dead pipeline. |

> **In one sentence:** spin a thousand random trading rules on a tape where nothing is real, and the luckiest one will look "statistically significant" by every naive measure — which is why a backtest *t*-stat means nothing until you correct for how many rules you tried.

## What we tested

The most dangerous number in quantitative finance is the backtest Sharpe of the rule you
*kept*. Bailey, López de Prado and co-authors proved that with enough trials you can hit a
Sharpe of 2 on pure noise; Sullivan-Timmermann-White ran ~7,800 technical rules and showed
the "best" ones evaporate under White's Reality Check. We build the p-hacking machine
directly: generate **N = 1,000 random rules** (a feature, a comparison, a threshold), backtest
every one honestly (one execution lag, excess-of-cash, costs), and study the distribution of
the *best* one under a controlled **null** (no edge), a **positive control** (a planted edge
the harness must rediscover), and the **real** SPY tape — with the multiple-testing reckoning
(Bonferroni + Reality Check) that the headline never shows. (Distinct from
[Study 350](../../350-dartboard-portfolio/), which randomises *holdings*; here we randomise the
*rule*.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the slot-machine intuition, why the luckiest of 1,000 rules always looks brilliant, and the one question that breaks the spell |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the null distribution of best-of-N, HAC *t* vs Bonferroni vs White's Reality Check, the planted-edge positive control, costs and out-of-sample decay |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`data_mining_roulette/`](data_mining_roulette/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
