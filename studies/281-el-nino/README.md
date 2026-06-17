# Study 281 -- El-Nino

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does El Nino / La Nina move equities or commodities?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | El-Nino-minus-La-Nina S&P gap **+3.2pp**, Welch t = **+0.69**, HAC t = **+0.44**, one-way ANOVA p = **0.60**, permutation p = **0.48**; oil even weaker and wrong-signed. n ~ 25/phase can't resolve a weather premium. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-El-Nino / short-La-Nina loses money gross (**-0.6%/yr** vs **+8.1%/yr** buy-and-hold); shorting La Nina years discards the equity premium. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The genuine ENSO-weather link does not transmit to prices; the tradable oil ETF moved the *opposite* way to the drought-spike folklore. |

> **In one sentence:** El Nino genuinely reshapes the weather and even spot commodity prices, but tested against the honest baseline (the S&P drifts up 73% of years anyway), the ENSO phase carries no tradable signal in equities or oil -- a real physical effect that never reaches the tape.

## What we tested

We hardcode NOAA's Oceanic Nino Index (ONI, NDJ season) for every winter 1950-2024 in
`data.py`, classify each into El Nino / La Nina / Neutral (the +/-0.5 threshold), and
join it -- at a one-year execution lag, so the trade is actionable -- to ^GSPC and USO
(crude oil) calendar-year returns. We compare per-phase mean returns against the
**unconditional baseline** (the S&P's 73% up-rate, +9.4% mean), and test with a Welch
t (El Nino vs La Nina), a one-way ANOVA across all three phases, a 10,000-shuffle
permutation test on the mean gap, a Newey-West HAC t on the long-short return series,
and a Bonferroni correction across the equity/oil tapes. The synthetic positive
control confirms the machinery finds a planted ENSO premium; the real tape confirms
there is none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the ENSO cycle, the per-phase means, why the long-short loses, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Welch t, one-way ANOVA, permutation distribution, HAC t, the oil tape, multiple comparisons, the n~25 power wall |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`el_nino/`](el_nino/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
