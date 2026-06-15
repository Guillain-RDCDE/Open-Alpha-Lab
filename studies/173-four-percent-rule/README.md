# Study 173 — Four-Percent-Rule

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | 100% historical US survival at 4% across 122 rolling 30-year cohorts (Shiller 1872–2022); SAFEMAX 4.14%; 95%-confidence SWR 4.49%. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Worst cohort (1966) ends at 0.13× initial wealth; Q4 CAPE at retirement → median terminal wealth 0.74×; low-yield world SWR collapses to ~2%. |
| **Survives a long life?** | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | At 40-year horizon, 95%-SWR drops to 3.97%; historical survival at 4% already only 94.6%. |

> **In one sentence:** Bengen's 4% rule is historically validated on US data — a 100% survival rate across 122 thirty-year windows — but it was calibrated on the most favourable equity market in recorded history, and sequence-of-returns risk plus today's elevated CAPE leave a much thinner forward margin of safety.

## What we tested

The famous 1994 Bengen recipe: a retiree withdraws 4% of initial portfolio value annually (inflation-adjusted, so a constant real dollar amount) from a 60% stock / 40% bond mix, rebalanced each year.  We run every available rolling 30-year window from the Shiller monthly dataset (1872–2022), giving 122 cohorts.  Then we binary-search for the true safe withdrawal rate (SAFEMAX = 4.14%; 95%-confidence = 4.49%), quantify sequence-of-returns risk (same mean, worst years first → terminal wealth halved), condition on starting CAPE (Q4 valuations slash the margin of safety without causing failure), extend to a 40-year horizon, and stress-test against a synthetic low-yield world (equity real 3% → SWR ~2%).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | survival charts, the near-miss of 1966, why order matters more than mean, and why 4% may be too generous today |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | SWR binary search, cohort terminal-wealth distribution, CAPE quartile conditioning, synthetic positive control, 40yr horizon |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`four_percent_rule/`](four_percent_rule/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
