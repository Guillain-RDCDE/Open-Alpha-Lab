# Study 53 — Jackpot 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do "lottery" (high-MAX) stocks underperform? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On large caps it runs **backwards**: the textbook long-low/short-high trade *lost* **−10.4%/yr (Sharpe −0.49, Lo t −2.5)**. High-MAX names beat the placid ones. |
| **Tradability** — is there a lottery premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross, and the inversion *strengthened* (Sharpe −0.40 → −0.71) as high-vol growth survivors led the 2010s. |
| **"Lottery effect on tradable large caps"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The premium lives in small/micro caps; survivorship + the growth regime erase and reverse it on liquid stocks. |

> **In one sentence:** the lottery (MAX) effect — high single-day-gain stocks should underperform — not only fails on tradable large caps but *inverts*: the high-MAX, high-volatility survivors beat the calm names, significantly and increasingly, because the real behavioural premium hides in the micro-caps you can't easily trade.

## What we tested

The **lottery / MAX effect** (Bali, Cakici & Whitelaw 2011): investors overpay for stocks with recent jackpot-like daily gains, so high-MAX names subsequently underperform — the textbook trade is long low-MAX, short high-MAX. We rank **436 S&P 500 names** each month by MAX (the mean of their five highest daily returns over the past month) and measure the long-short, its sign and significance, and its evolution over time. The offline control is a synthetic daily panel where high-MAX (high-volatility) stocks genuinely underperform (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "avoid the lottery stocks" backfired on large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the textbook hedge with its Lo t-stat, the strengthening inversion, the micro-cap/survivorship/regime explanation |

The fingerprinted real-data run (436 names, 2000–2026, fp `ed307fe3bd8b`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic panel in [jackpot/data.py](jackpot/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
