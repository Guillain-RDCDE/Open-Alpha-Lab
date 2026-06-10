# Study 30 — House-Edge 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the risk management actually work? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The vol-targeted, trend-gated dip-buyer **halves the drawdown**: max-DD **−27.8%** vs buy-and-hold's **−55.7%** on the S&P 500 (1990–2026), and **−35%** vs **−44%** on the synthetic control. Real, robust risk control. |
| **Tradability** — does the *levered* book beat buy-and-hold once financed honestly? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. Honest full-notional financing → CAGR **5.3%** vs the total-return index's **10.6%**, at a *lower* Sharpe (**0.42** vs **0.65**) and an **identical Calmar (0.19)**. It never beats buy-and-hold at *any* financing cost. You buy drawdown insurance — and pay for it 1-for-1 in return. |
| **Free leverage?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The optimistic accounting that finances only the slice *above* 100% (and ignores dividends) hides **1.55 pts/yr** of cost vs charging the full notional. That gap — the broker's house edge on your leverage — is what flatters levered-timing backtests. |

> **In one sentence:** a vol-targeted contrarian dip-buyer genuinely *halves your drawdown*, but once you charge leverage its real, full-notional financing cost it makes **half the return** of simply holding a total-return index — the "lever a good signal to beat the market" pitch is a mirage, and the free-leverage assumption that sells it is busted.

## What we tested

Every levered-ETF and CFD timing product makes the same pitch: *take a smart timing model — buy the dips, target volatility, gate on the trend — add leverage, and you beat buy-and-hold.* Leverage, the story goes, just amplifies a good signal. We build a faithful version of that strategy (RSI(2) contrarian entries, `target_vol / realised_vol` sizing, a 200-day trend gate) and run it on the S&P 500 back to 1990 — but we charge leverage what it **actually** costs: the *entire* notional financed at the T-bill rate plus a broker markup, every day, with dividends credited and idle cash earning interest. The control is a synthetic GARCH-with-bear-regimes index where the same machinery is exercised offline; the headline run is the real tape.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why levered timing *feels* like a free lunch, the drawdown it really buys you, and the financing bill the brochure leaves off |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | idealized vs full-notional cost accounting, the financing-markup sweep, drawdown-vs-return decomposition, the house-edge identity |

The fingerprinted real-data run (S&P 500, 1990–2026) is in [docs/results.md](docs/results.md); reproduce it offline on the synthetic control via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py), or on the real tape via [examples/verify.py](examples/verify.py) (`--fetch` to download ^GSPC + ^IRX).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
