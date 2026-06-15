# Study 178 — CCI

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Breach-entry gross **−33.11 bps/trade**, HAC *t* = **−2.06** (wrong sign); actively *worse* than a random-direction control (Δ = −28.73 bps). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross expectancy significantly negative → a loser before costs at every hold window tested (1–20 days). |
| **Beats a coin?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | 45.1% win-rate (below 50%); CCI extremes predict the *wrong* direction on 2016–2026 US equity data. |

> **In one sentence:** Lambert's CCI is a mean-reversion oscillator built for commodity cycles — on trending US equity daily bars, its ±100 thresholds act as a momentum-continuation signal in disguise, and buying "oversold" loses significantly more than a fair coin.

## What we tested

Donald Lambert's 1980 Commodity Channel Index measures how far the typical price strays from its 20-period moving average, scaled by mean absolute deviation. The folk rule: buy when CCI crosses below −100 ("oversold"), sell short when it crosses above +100 ("overbought"). We test two framings — breach entry (first bar inside the zone) and cross-back entry (confirmed exit from the zone) — across six liquid daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, 10 years), with hold windows from 1 to 20 days, pinned against a **random-direction control** on identical entries. A synthetic tape with tunable mean-reversion serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what CCI actually measures, why "oversold" on equity momentum is often a trap, the fair-coin comparison |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, hold-period sweep, cost sensitivity, bootstrap Sharpe CI, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cci/`](cci/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
