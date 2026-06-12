# Study 70 — Digital-Gold ₿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is bitcoin a gold-like store of value / haven? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No: ~uncorrelated with gold (**+0.09**), correlation with stocks **rising +0.13→+0.36**, and in equity crashes it fell **harder than stocks (−18% vs −10%)**, up in **0%** of them. |
| **Tradability** — did a small sleeve help a portfolio? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Yes historically (5% BTC: Sharpe 0.82→0.95) — but purely on a **+64%/yr, −83%-drawdown** trend. A leveraged return bet, not a hedge. |
| **"Bitcoin is digital gold"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | It's a high-beta risk asset, the opposite of a haven when it matters. |

> **In one sentence:** bitcoin isn't digital gold — it's near-uncorrelated with gold, increasingly correlated with stocks, and in crashes it falls *harder* than equities and never rallies; it's been a spectacular high-beta *return* asset (so a small sleeve flattered Sharpe), but that's a drawdown-heavy growth bet, not the safe-haven store of value it's sold as.

## What we tested

The **"digital gold" thesis**: that bitcoin is a gold-like store of value and crisis hedge. We test it on daily **BTC-USD / SPY / GLD**, 2014–2026 — BTC's correlation with gold and with stocks (and how that correlation has *drifted*), its return in equity-crash months (does it hold/rise like a haven, or fall like a risk asset?), and the effect of adding a small BTC sleeve to a stock portfolio. The offline control is a synthetic world where the bitcoin-like asset either loads on the stock factor (a risk asset) or stays uncorrelated (a true digital gold).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a thing that crashes with stocks isn't gold — and what the sleeve really is |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the correlations & their drift, the crash-month behaviour, the sleeve overlay |

The fingerprinted real-data run (BTC/SPY/GLD, 2014–2026, fp `6e2e2a7c7c98`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads the BTC/SPY/GLD pull); the offline machinery proof runs on the synthetic world in [digital_gold/data.py](digital_gold/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
