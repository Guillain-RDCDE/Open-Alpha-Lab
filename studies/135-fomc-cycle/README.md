# Study 135 — FOMC-Cycle

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Even-week SPY mean **+6.58 bps/day** (HAC *t* = 3.94) but the direct even-minus-odd Welch *t* = **1.34**; placebo p = 0.11; out-of-sample (post-2019) gap is **−1.72 bps** (*t* = −0.30). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even-week premium vs buy-and-hold is not statistically significant (t = 1.34); post-2019 the live strategy *underperformed* buy-and-hold by ~4.3%/yr. |
| **Post-publication decay** | ![Confirmed](https://img.shields.io/badge/Decay--Confirmed-8b949e?style=flat-square) | Pre-2019 gap: **+5.07 bps** (*t* = 1.71). Post-2019 gap: **−1.72 bps** (*t* = −0.30). Consistent with McLean-Pontiff (2016). |

> **In one sentence:** the FOMC even-week premium documented by Cieslak, Morse & Vuolteenaho (2019) showed a historically elevated even-week return (+5 bps/day pre-2019) that has since reversed to flat-to-negative post-publication, making it a textbook case of academic arbitrage destroying an anomaly before it could be traded profitably.

## What we tested

Cieslak, Morse & Vuolteenaho (2019, *Journal of Finance* 74:5) documented that US equity returns since 1994 accrue disproportionately in the even-numbered weeks (0, 2, 4) of the ~6-week FOMC inter-meeting cycle. Week 0 starts on the FOMC statement day; each cycle week covers 5 trading days. We replicate the test on SPY daily total returns from 1994 to 2026 using the published FOMC meeting schedule, assign each trading day its cycle-week label (no look-ahead), and measure even-vs-odd mean return with a Welch t-test plus a 500-seed random-permutation placebo. We then split the sample at 2019-01-01 (the publication year) to test post-publication decay, and sweep the cost of a biweekly-switching long strategy. A deterministic synthetic tape with a tunable even-week premium serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the FOMC cycle in plain language, the pre-2019 evidence, the post-publication reversal, why the premium is largely just equity risk |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | week-by-week HAC t-stats, Welch t on the gap, 500-seed permutation placebo, rolling Sharpe decay, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fomc_cycle/`](fomc_cycle/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
