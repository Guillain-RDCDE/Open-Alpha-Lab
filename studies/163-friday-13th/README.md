# Study 163 — Friday-13th

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Friday-13th mean **+6.90 bps/day** vs +4.60 bps for other Fridays; contrast +2.30 bps, Welch p = **0.82**, HAC t = +0.57. Bonferroni-corrected p across the four "special Friday" slots = **1.00**. The 13th is the *least* anomalous of the four. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No directional signal exists to trade. At ~1.7 events/year any cost trivially exceeds the noise-level premium. |
| **Superstition vs data** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On the S&P 500 since 1928, Friday-13th is a slightly *above*-average day. The fear points the wrong direction. |

> **In one sentence:** Kolb & Rodriguez (1987) found a negative Friday-13th anomaly in the 1940-1987 DJIA; on the S&P 500 over 99 years the 13th is indistinguishable from any other Friday — and if anything slightly *better* than average — a superstition with the wrong sign.

## What we tested

The academic claim (Kolb & Rodriguez 1987): daily returns on Friday the 13th are systematically *lower* than on other Fridays, as if investor superstition depresses prices. We test it on **^GSPC daily returns from 1928-01-03 to 2026-06-12** (168 Friday-13ths, ~1.7/yr), with three honest controls: (1) all other Fridays, (2) Friday-the-6th as a matched placebo (same weekday, no folklore), and (3) a full **Bonferroni-corrected** sweep across all four "special Friday" day-of-month slots (6, 13, 20, 27) — the multiple-comparisons kill shot that any superstition-mining exercise must survive.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the superstition, the win-rate for the 13th, the placebo reveal, why multiple comparisons doom it, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Welch tests, Bonferroni table, sub-period K&R era, synthetic positive control, power analysis |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`friday_13th/`](friday_13th/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
