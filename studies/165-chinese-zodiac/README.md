# Study 165 -- Chinese-Zodiac

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pre-CNY FXI HAC t = **+0.04** (mean +0.05% vs baseline -0.23%). Dragon vs rest: Welch t = +0.84, Bonferroni-p = 1.00. Permutation: 24% of random animals match Dragon. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Sub-0.1% gross pre-CNY return goes negative at any realistic spread. No zodiac-year signal to trade. |
| **Lucky Dragon?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | n = 2 Dragon years on FXI. 12 x ~3 years is structurally underpowered for a 12-way test. Min detectable effect at 80% power ~+100%/yr; observed Dragon premium +18%/yr. |

> **In one sentence:** the Chinese zodiac market folklore is a textbook small-sample mirage -- 12 animals times ~3 years is structurally too thin to test a 12-way hypothesis, the pre-CNY rally is noise on both FXI and the S&P 500, and a Bonferroni correction buries even the most extreme raw t-stat.

## What we tested

Chinese New Year market folklore comes in two flavours: (A) a **pre-CNY 10-day rally** in Chinese/Asian equities driven by festive optimism and portfolio positioning; and (B) a **zodiac-year cycle** with Dragon years reliably bullish and certain other animals bearish. We hardcoded the Chinese New Year dates and zodiac animals for 1990-2026, ran both tests on FXI (primary) and ^GSPC (secondary), applied a **12-way Bonferroni correction** on the zodiac-year test, and quantified the minimum detectable effect for the effective n per animal (~2-3 years). Both tests fail. The pre-CNY window yields HAC t = +0.04 on FXI; the Dragon-year Bonferroni-corrected p = 1.00. A power analysis shows that detecting a genuine 20%/year Dragon premium would require ~50 Dragon years, or 600 calendar years of data.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the zodiac table, why 12 x 3 is not enough, the Bonferroni reckoning in plain language, the cost math |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC inference on pre-CNY windows, 12-way Bonferroni table, power analysis (MDE ~100%/yr at n=3), permutation null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`chinese_zodiac/`](chinese_zodiac/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
