# Study 161 — Year-Ending-Five

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Digit-5 mean +19.3% vs grand mean +6.4%, HAC *t* = **+3.21**; best-of-10 permutation p = **0.008** — clears the 5% bar, but n = 16 per bucket and no mechanism. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timing advantage of avoiding digit-0 years is **+0.17%/yr** over 154 years — below any realistic estimation noise; a long-only investor simply holds through. |
| **Myth-check: digit-5 really IS the best bucket?** | ![Partially_Real](https://img.shields.io/badge/Myth--check-Partially_Real-8b949e?style=flat-square) | Arithmetically correct in the 1872–2025 Shiller sample; holds pre- and post-1945. But n = 16 + no mechanism + post-hoc selection = the textbook definition of pattern-in-noise. |

> **In one sentence:** years ending in 5 genuinely topped the decennial table in 154 years of S&P 500 history (+19.3% average vs +6.4% grand mean), but 16 observations with no causal story and a ×10 multiple-comparisons tax is not a signal — it is a very compelling coincidence.

## What we tested

The Stock Trader's Almanac mnemonic: *"years ending in 5 are the best for stocks; years ending in 0 are the worst — just look at the last century."* We take it literally: on the Shiller S&P 500 monthly dataset (1872–2025, n = 154 annual returns), we group calendar-year returns by last digit, measure the digit-5 excess with a HAC t-stat, and then run a 50,000-iteration permutation test to answer the real question: given that we are picking the best of 10 digits post-hoc, what is the data-snooping-corrected probability of seeing a digit this dominant by chance? We also check whether the pattern holds pre- and post-1945, and compute how much a "buy except in digit-0 years" timer actually earns.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the decennial wheel in plain language, the full digit table, what "best-of-10 by luck" looks like, why the tiny timing edge is unactionable |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, best-of-10 permutation correction, pre/post split, the positive control showing the engine works, the "10 simultaneous tests" inflation anatomy |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`year_ending_five/`](year_ending_five/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
