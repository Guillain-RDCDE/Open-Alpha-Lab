# Study 58 — Bunker 🛡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the min-vol ETF deliver the low-vol anomaly? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Half. USMV genuinely cut risk (vol **11.4% vs 14.3%**, drawdown **−20% vs −24%**) but its Sharpe (**0.99**) sat *below* the market's (**1.05**) over 2011–2026. |
| **Tradability** — is it a market-beater? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | A fine defensive holding, but it trailed on return (**+11.4% vs +14.8%/yr**, spread −3.5%/yr) — the promised free Sharpe didn't show in a bull sample. |
| **"Delivers the low-vol Sharpe edge"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The anomaly's signature (higher return *per unit of risk*) is absent here; min-vol's edge lives in bear markets and full cycles. |

> **In one sentence:** the minimum-volatility ETF does exactly what it says for *risk* — 20% less volatility and a shallower drawdown — but over 2011–2026 it did not beat the market risk-adjusted (Sharpe 0.99 vs 1.05), so the low-vol anomaly's promised free Sharpe is a fine defensive bunker, not a market-beater in a bull-dominated sample.

## What we tested

The **low-volatility anomaly** as an investable product: does the min-vol ETF (**USMV**) deliver the better risk-adjusted returns the academic effect ([Study 18 Dull-Roar](../18-dull-roar/)) promises? We compare USMV to **SPY** over USMV's life (2011–2026), leg by leg — CAGR, Sharpe, volatility, drawdown — plus the spread and the volatility-reduction ratio. The offline control is a synthetic world where the min-vol sleeve has beta<1 (lower vol) and a tunable low-vol alpha (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a calmer ride didn't mean a better risk-adjusted return here |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the leg comparison, the vol-reduction ratio, the spread, the bull-regime caveat |

The fingerprinted real-data run (USMV vs SPY, 2011–2026, fp `59a4a76cab23`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [bunker/data.py](bunker/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
