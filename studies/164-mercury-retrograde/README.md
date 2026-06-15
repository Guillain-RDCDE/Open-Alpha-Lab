# Study 164 — Mercury-Retrograde

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pooled Welch *t* = **−2.21** (nominally crosses the bar) but HAC *t* on retrograde-only mean = **−1.17**; the 2010–2019 decade shows **zero effect** (Welch *t* = −0.11); strip the COVID-crash retrograde coincidence and *t* drops to −1.87. Fragile in every robustness check. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Retrograde-avoidance earns +2.52% excess CAGR **gross**, with a HAC *t* of only **+1.13** — not statistically significant, and any realistic friction erases it. |
| **The planets do not trade** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The borderline signal is a regime coincidence: the dot-com bust, GFC, and COVID crash overlap by chance with specific retrograde windows. The 2010–2019 bull-market decade shows no effect whatsoever. Mercury's apparent backwards motion does not move index prices. |

> **In one sentence:** Mercury retrograde's market curse produces a nominally borderline Welch *t* only because three major crashes happened to land near retrograde windows — in the calm 2010–2019 decade the effect is zero, the HAC *t* on retrograde returns is −1.17, and the avoidance strategy's excess is statistically invisible (*t* = 1.13). The planets do not trade.

## What we tested

Financial astrology's most popular bad-omen: Mercury in retrograde (~3 periods/year, ~3 weeks each, ~19.7% of calendar days) supposedly brings market chaos, falling prices, and wild volatility. The retrograde dates (2000–2026) are hardcoded from the NASA JPL ephemeris. On ^GSPC daily returns we run four tests: **(A)** mean return during retrograde vs direct days with HAC and Welch *t*; **(B)** realised volatility comparison (Levene test); **(C)** retrograde-avoidance strategy vs buy-and-hold; **(D)** 5,000-permutation null to ask how often a randomly chosen ~19.7% of days looks as bad. A deterministic synthetic tape with a tunable retrograde drag serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the borderline number, why it falls apart decade by decade, and why a crisis coincidence is not a planetary force |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*, Welch *t*, Levene variance ratio, sub-period breakdown, COVID-strip robustness, permutation null, avoidance strategy with full inference |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mercury_retrograde/`](mercury_retrograde/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
