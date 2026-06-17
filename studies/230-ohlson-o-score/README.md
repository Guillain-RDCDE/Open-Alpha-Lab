# Study 230 — Ohlson O-score

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hedge (safe−distressed) = **−4.45%/yr**, HAC *t* = **−1.59**; firm-level corr(O, return) = **+0.048**. Below the ±2 inference bar. Survivorship-biased upper bound. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Noisy annual hedge, heavy survivorship haircut, annual rebalance costs; no implementable edge. |
| **Does the O-score beat Altman-Z at pricing distress?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | O-score leans *distress-premium* (high-O earns +26%/yr vs +21.5% for safe), opposite direction to Altman-Z (study 123), yet both fail |t|≥2. Neither model reliably prices distress risk. |

> **In one sentence:** Ohlson's nine-variable logit distress model is no better than Altman's five-variable Z-score at predicting equity returns — the HAC t-stat of −1.59 barely misses the bar, and survivorship bias inflates the distressed-bucket return, masking the true picture.

## What we tested

> *Does the Ohlson O-score price distress any better than Altman-Z did?*

The Ohlson (1980) O-score is a logit bankruptcy predictor that maps nine
accounting ratios into a probability of financial failure:

O = −1.32 − 0.407·SIZE + 6.03·TLTA − 1.43·WCTA + 0.076·CLCA
    − 1.72·OENEG − 2.37·NITA − 1.83·FUTL + 0.285·INTWO − 0.521·CHIN

Higher O = higher distress probability. We test whether O-score rank-sorts
2008–2023 S&P 500 equity returns (EDGAR 10-K data, one-year lag) using the
same protocol as the companion study 123 (Altman-Z). The hedge sorts: long
low-O (safe) / short high-O (distressed). Survivorship bias is explicit —
all results are upper bounds.

Key result: the distressed bucket (high-O) earns **+26.0%/yr** vs **+21.5%**
for the safe bucket — but the HAC t-stat of −1.59 falls short of the ±2
inference bar, and the bootstrapped Sharpe CI [−1.07, +0.02] straddles zero.
No reliable signal.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the O-score formula explained, the distress-premium vs distress-puzzle debate, O-bucket return chart, why it doesn't clear the bar |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | annual bucket table, HAC t-stat, bootstrap Sharpe CI, firm-level cross-section, survivorship-bias anatomy, synthetic positive control, Altman-Z comparison |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ohlson_o_score/`](ohlson_o_score/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
