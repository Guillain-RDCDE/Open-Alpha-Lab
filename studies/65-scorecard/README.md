# Study 65 — Scorecard 📋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high F-score firms beat low ones? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Not on large caps — it *inverts*. Long high-F / short low-F lost **−3.7%/yr (Sharpe −0.39, t −1.7, 33% hit)**: the low-F leg (+22.2%) *beat* the high-F leg (+18.5%). |
| **Tradability** — is there an F-score premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Wrong sign on the S&P 500: high-F is crowded and priced, low-F snaps back. |
| **"F-score sorts winners on tradable large caps"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Piotroski built it *within small-cap value*; the large-cap universe is the wrong one. |

> **In one sentence:** Piotroski's celebrated nine-point F-score doesn't sort returns on the S&P 500 — long high-F / short low-F lost −3.7%/yr (t −1.7) and *inverted*, because the screen's edge is a small-cap, high-book-to-market phenomenon and large-cap high-F is already fully priced.

## What we tested

The **Piotroski F-score** (2000): a nine-point checklist of fundamental health (profitability, leverage / liquidity, operating efficiency) where high-score firms (8–9) are meant to beat low-score firms (0–2). We build the score from **real SEC EDGAR data** — nine `us-gaap` concepts for ~237 current S&P 500 members — sort each year, go long the top-tercile (healthy) and short the bottom-tercile, and measure the annual hedge. We flag the limits openly: short XBRL window (~2009+) and a survivorship-biased large-cap universe — the *opposite* of the small-cap value universe Piotroski tested. The offline control is a synthetic firm panel with a known F-score premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a strong balance sheet doesn't predict large-cap winners |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long-short, the inverted sign, the small-cap / value / crowding explanation |

The fingerprinted real-data run (~237 names, ~2009–2025, fp `e3884e329059`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads the shared EDGAR pull); the offline machinery proof runs on the synthetic panel in [scorecard/data.py](scorecard/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
