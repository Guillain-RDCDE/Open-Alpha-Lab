# Study 64 — Share-Shuffle 🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do share issuers underperform buyback firms? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Not on large caps — it *inverts*. The textbook long-buyback/short-issuer trade lost **−2.8%/yr (Sharpe −0.30, t −1.3, 28% hit)**: issuers (+22.8%) *beat* buybacks (+20.0%). |
| **Tradability** — is there an issuance premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Wrong sign on liquid names: the heavy issuers were the growth/tech winners diluting via stock comp and M&A. |
| **"Issuance anomaly on tradable large caps"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The real effect lives in small caps (Pontiff-Woodgate); survivorship + the growth regime invert it here. |

> **In one sentence:** the net-issuance anomaly — issuers underperform, buybacks outperform — runs backwards on tradable large caps (−2.8%/yr, t −1.3), because the heaviest issuers were the growth winners diluting through stock-based comp; the real premium hides in small caps, the wrong universe for a free EDGAR pull.

## What we tested

The **net-share-issuance anomaly** (Pontiff & Woodgate 2008; Daniel & Titman 2006): firms that issue shares subsequently underperform, firms that buy back outperform. We build it on **real SEC EDGAR data** — the year-on-year change in diluted shares outstanding for ~412 current S&P 500 members — sort by net issuance, go long the buyback/low-issuance names and short the high-issuance ones, and measure the annual hedge. We flag the limits openly: short XBRL window (~2010+) and a survivorship-biased large-cap universe. The offline control is a synthetic firm panel with a known issuance premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the disciplined buyback firms" lost to the diluters on large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long-short, the backwards sign, the small-cap / survivorship / growth-regime explanation |

The fingerprinted real-data run (~412 names, ~2010–2025, fp `46ca77dffb10`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads the shared EDGAR pull); the offline machinery proof runs on the synthetic panel in [share_shuffle/data.py](share_shuffle/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
