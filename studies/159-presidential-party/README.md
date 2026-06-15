# Study 159 — Presidential-Party

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Raw gap **+4.6%/yr** real (1927-2023); Welch *t* = **+1.45** on monthly obs; block-permutation *p* = **0.32** on 17 terms. Not certified — the Depression/WWII coincidence under FDR does most of the lifting. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Party is known only post-election (already priced); one data point per 4-year term; no advance signal exists on which to trade. |
| **Tiny-n?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | 8 D terms, 9 R terms since 1927 — the inference bar is set at the term level (~17 draws), not the monthly level (1,160 obs). On 17 draws no gap is certifiable. |

> **In one sentence:** stocks did earn ~4.6%/yr more per year in real terms under Democrats than Republicans since 1927, but with only 17 presidential terms to count the effect is statistically indistinguishable from business-cycle luck — and knowing the winner after the election is too late to trade it.

## What we tested

The famous finding of Santa-Clara & Valkanov (2003, *Journal of Finance*): real S&P 500 returns are dramatically higher under Democratic presidents. We test it on Shiller's monthly real-price series (1927-2023, *n* = 1,160 months) with the presidents hardcoded in `data.py`, using three escalating inference tools — Welch t on monthly observations, Newey-West HAC t on a party-dummy regression, and a **block-permutation test at the term level** (the only test that respects where the real degrees of freedom live). We also run post-war and pre-war sub-samples to isolate the Depression/WWII confound.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the raw gap in plain language, the permutation test, why the Depression skews everything, why you can't trade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-party HAC t-stats, the three inference tests, sub-sample robustness table, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`presidential_party/`](presidential_party/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
