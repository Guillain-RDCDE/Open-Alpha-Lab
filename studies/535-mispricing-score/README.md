# Study 535 — Mispricing-Score

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Does averaging eleven anomalies into one composite "mispricing score" — and shorting the most overpriced names — beat any single anomaly?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the composite earn a real long-short edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The score genuinely *sorts* (label-shuffle placebo **p = 0.007**), but the long-short SYY specify earns **no positive** *t* ≥ 2 — it earns a *negative* one: **−7.68%/yr gross (HAC t = −1.84)**, **−9.50%/yr net (t = −2.28)**. No real positive edge on this tape. |
| **Tradability** — does the spread pay after costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Loses gross **and** net, Sharpe **−0.47**, max drawdown **−80%**. Costs *deepen* the loss. The only salvageable residue is the long (cheap) leg as a long-only tilt (+15.45%/yr, *t* +4.26) — itself a survivor mega-cap artefact. |
| **"Is the edge in the short leg?" (SYY's headline)** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The opposite. On survivors the "overpriced" glamour names (NVDA/AAPL cohort) are the decade's *winners*; shorting them loses **−23.1%/yr (t = −5.04)**. The failed overpriced names that would make the short leg pay have **delisted**. |

> **In one sentence:** the Stambaugh-Yu-Yuan composite mispricing score *does* carry real cross-sectional information (placebo p = 0.007), but on a 45-name large-cap **survivor** basket its short leg is exactly inverted — the "overpriced" names are the surviving mega-cap winners, so the long-short loses −7.68%/yr gross and −9.50% net, and SYY's signature short-leg edge is Busted by the very delisting bias the paper warned a clean test must avoid.

## What we tested

Stambaugh, Yu & Yuan (2015): average the cross-sectional ranks of well-known anomalies into one
**composite mispricing score** (high = overpriced), then long the cheapest quintile and short the
most overpriced. We build the **six price-computable** members of the eleven-anomaly set — 12-1
momentum, MAX/lottery, 6-month volatility, 12-month price run-up (asset-growth proxy), distance to
the 52-week high, short-term reversal — because the fundamental members (accruals, net issuance,
profitability, distress) need clean per-name financials yfinance does not deliver reliably; we
document that limitation rather than fake it. Panel: 45 large-cap S&P 500 names, yfinance daily
prices 2010–2025 (178 monthly observations), one-day execution lag, 10 bps/side + 75 bps/yr borrow.
A deterministic synthetic positive control (planted, state-dependent mispricing premium) proves the
engine is faithful. *Distinct from the single-anomaly studies it composes —
[238 Betting-Against-Beta](../238-betting-against-beta/), [330 Low-Volatility](../330-low-volatility-anomaly/),
[231 Sloan-Accruals](../231-sloan-accruals/), [244 Asset-Growth](../244-asset-growth/) — this is the
**meta-anomaly** that claims to beat them all.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why averaging anomalies should help, why the short leg is supposed to drive it, and why a survivor basket flips the whole thing on its head |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the composite rank construction, the placebo p = 0.007, the long-vs-short leg split with HAC *t*, costs + borrow, the cutoff-robustness ladder, and the synthetic positive control |

The fingerprinted real-data run (45 S&P 500 large-caps, 2010–2025, fp `b6d4a14d508b`) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the synthetic world in
[`mispricing_score/data.py`](mispricing_score/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`mispricing_score/`](mispricing_score/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
