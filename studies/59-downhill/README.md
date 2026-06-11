# Study 59 — Downhill ⛷️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a term premium? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes. Intermediate Treasuries (IEF) beat cash (BIL) by **+2.2%/yr** over 2003–2026 — longer bonds do pay more than bills. |
| **Tradability** — should you ride the curve for it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The excess earns a Sharpe of just **0.32**, and stretching to 7–10y bought a **−23%** drawdown in 2022. Cash's Sharpe was **1.82**, 1–3y's **1.13**. |
| **"Is the duration risk worth it"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The Sharpe *falls* as you extend duration — you're paid for it, just nowhere near enough. |

> **In one sentence:** the term premium is real — longer Treasuries beat cash by ~2%/yr — but riding the curve for it is a poorly-compensated trade: the excess earns a Sharpe of 0.32 against cash's 1.82, and reaching for duration carved a −23% drawdown in 2022, so the risk-adjusted numbers say leave it on the table.

## What we tested

The classic fixed-income trade of **riding the yield curve** — holding longer-duration Treasuries to harvest the **term premium** and roll-down. We compare three points on the curve — **IEF** (7–10y), **SHY** (1–3y), **BIL** (cash) — over 2003–2026: each one's CAGR, Sharpe, volatility and drawdown, and the isolated term premium (IEF − BIL) with its t-stat. The question isn't whether longer bonds earn more (they do) but whether the *extra* return justifies the *extra* risk. The offline control is a synthetic Treasury world with a real term premium and duration volatility (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "longer bonds pay more" is true and still a bad risk-adjusted trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the duration ladder (return up, Sharpe down), the isolated term premium, the 2022 drawdown |

The fingerprinted real-data run (IEF/SHY/BIL, 2003–2026, fp `e3c760f99f78`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [downhill/data.py](downhill/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
