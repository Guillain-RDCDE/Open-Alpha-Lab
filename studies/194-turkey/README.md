# Study 194 — Turkey

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Black Friday (day after Thanksgiving) mean **+20.62 bps/day** vs +4.36 bps for other Fridays; contrast **+16.26 bps**, HAC t = +2.01, but Welch p = **0.10** and Bonferroni-corrected p = **0.20** (k=2). Wednesday-before is flat (contrast −1.30 bps, p = 0.91). With only 97–98 events in 99 years the study is genuinely underpowered. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~1 event/year at ~16 bps/event = ~16 bps/yr gross. Any realistic execution cost consumes a large fraction of that, and the uncertainty on the 16 bps estimate (CI ±~20 bps) makes the strategy untrustworthy. |
| **Pre-holiday vs post-holiday** | ![Post--holiday_only](https://img.shields.io/badge/Pre--holiday_only-8b949e?style=flat-square) | The Black Friday half-day is the stronger side; the pre-holiday Wednesday is a clear null. The pattern is asymmetric and limited to the short post-holiday session. |

> **In one sentence:** The Wednesday before Thanksgiving shows no edge (−1.30 bps, p = 0.91); Black Friday shows a large but statistically borderline positive return (+16.26 bps, Welch p = 0.10, Bonferroni p = 0.20) that cannot be certified with only 97 observations — an intriguing breadcrumb, not a tradable edge.

## What we tested

The academic claim: the day before Thanksgiving (pre-holiday effect) and the half-day
after (Black Friday) deliver reliably positive S&P 500 returns. We test both on
**^GSPC daily returns from 1928-01-03 to 2026-06-12** (~98 Thanksgiving Wednesdays,
~97 Black Fridays), each compared against all other days of the same weekday (other
Wednesdays and other Fridays respectively). Controls: (1) matched-weekday baselines,
(2) Tuesday-before-Thanksgiving as a matched-window placebo, (3) Bonferroni correction
for two simultaneous primary tests (k=2, threshold p < 0.025).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Thanksgiving legend, the two-day teardown, the placebo reveal, why the signal is borderline at best, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Welch tests, Bonferroni correction, power analysis, synthetic positive control, sub-period check |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`turkey/`](turkey/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
