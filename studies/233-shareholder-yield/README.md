# Study 233 — Shareholder-Yield

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does high total shareholder yield predict outperformance? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Inverted on large caps. High-yield quintile (+20.0%) trailed low-yield (+22.8%): hedge **−2.8%/yr, Sharpe −0.30, t −1.3, 28% hit rate**. |
| **Tradability** — is there a yield premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Wrong sign. Low-yield growth names (low/zero buybacks, retained earnings) dominated 2008–2025. |
| **Is total shareholder yield the dividend factor done right?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Right theory, wrong universe. The composite metric doesn't recover a premium on S&P 500 survivors; growth beat yield throughout the post-2009 bull market. |

> **In one sentence:** shareholder yield (dividends + net buybacks) is the correct theoretical
> construct but produces a **−2.8%/yr, t −1.3** hedge on large-cap S&P survivors, because the
> low-yield cohort was dominated by the growth/tech winners that retained cash and drove the market
> — identical inversion to the pure net-issuance factor ([64 Share-Shuffle](../64-share-shuffle/)).

## What we tested

**Shareholder yield** = dividend yield + net buyback yield (year-on-year fractional decrease in
diluted shares outstanding). The claim (Faber 2007; Boudoukh et al. 2007): total capital return
predicts cross-sectional outperformance better than dividend yield alone — high yielders win, low
yielders lose. We test it on **SEC EDGAR diluted-share data** for ~412 current S&P 500 members
(2008–2025), sort annually by total yield, go long the top-20% and short the bottom-20%, and
measure the hedge vs next-year returns. Limits stated: survivorship-biased large-cap universe,
XBRL era only.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the high total-yield names" lost to the zero-yield growth cohort on large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long-short, the backwards sign, the growth-regime / survivorship explanation |

The fingerprinted real-data run (~412 names, 2008–2025, fp `f1b6fc02e713`) is in
[docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads
the shared EDGAR pull); offline machinery proof runs on the synthetic panel in
[shareholder_yield/data.py](shareholder_yield/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
