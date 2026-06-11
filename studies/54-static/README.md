# Study 54 — Static 📺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-idio-vol stocks earn less? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On large caps it runs **backwards**: the textbook long-low/short-high trade *lost* **−12.1%/yr (Sharpe −0.60, Lo t −3.1)**, steadily across the sample. |
| **Tradability** — is there a low-vol premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross and consistent (Sharpe −0.65 in both halves) — high-idio-vol growth survivors beat the calm names. |
| **"Idio-vol puzzle on tradable large caps"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The puzzle lives in small/micro caps; survivorship + the growth regime invert it on liquid stocks. |

> **In one sentence:** the idiosyncratic-volatility puzzle — high-vol stocks should earn *less* — inverts decisively on tradable large caps (−12%/yr, t −3.1), the near-twin of the MAX effect ([Jackpot](../53-jackpot/)), because idio-vol is a high-risk axis and the volatile growth survivors won the post-2009 market.

## What we tested

The **idiosyncratic-volatility puzzle** (Ang, Hodrick, Xing & Zhang 2006): stocks with high firm-specific volatility — the part of their return *not* explained by the market — subsequently earn *lower* returns, a famous puzzle since risk should pay. The textbook trade is long low-idio-vol, short high. We estimate idio-vol from a rolling 1-factor regression on **436 S&P 500 names**, run the long-short, and check its sign, significance and persistence. The offline control is a synthetic panel where high-idio-vol stocks genuinely underperform (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the calm stocks" lost to the volatile ones on large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the idio-vol hedge with its Lo t-stat, the steady inversion, the MAX twin, the micro-cap/survivorship story |

The fingerprinted real-data run (436 names, 2000–2026, fp `ed307fe3bd8b`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic panel in [static/data.py](static/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
