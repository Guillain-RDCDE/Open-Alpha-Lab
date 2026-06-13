# Study 113 — Gold-Silver-Ratio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Gross **+144 bps/trade**, HAC *t* = **+1.77** — below the \|*t*\| ≥ 2 bar on only **26 trades over 16 years** (1.6/yr); statistically inconclusive. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Costs are negligible (~1.6 trades/yr, ~78-day holds). The killer is extreme signal aridity and a 15-month estimated reversion half-life. |
| **Ratio mean-reverts?** | ![Not Supported](https://img.shields.io/badge/Not_Supported-8b949e?style=flat-square) | ADF p = 0.22 (fails to reject unit root); Engle-Granger p = 0.74 (not cointegrated); OU half-life = **301 trading days**. The structural premise is unconfirmed. |

> **In one sentence:** the gold/silver ratio wanders like a slow random walk (ADF p = 0.22, 301-day half-life, not cointegrated) — the z-score strategy earns a positive but inconclusive t = +1.77 on only 26 trades over 16 years, while a simple buy-and-hold of either metal delivers ~8%/yr with zero complexity.

## What we tested

The gold-to-silver ratio trade is one of the oldest commodity folk signals: when gold is historically expensive relative to silver (ratio extreme high), buy silver and sell gold, betting on a snap-back to the historical mean, and vice versa. We implement this as a z-score rotation between GLD and SLV ETFs (252-day rolling window, enter at |z| > 1.5, exit at |z| < 0.5), compare it against a **random-direction control** on identical entry/exit dates, test whether the ratio is actually stationary (ADF, Engle-Granger cointegration, OU half-life estimation), and benchmark against buy-and-hold of each metal. A deterministic synthetic OU pair with tunable mean-reversion speed serves as the positive control, confirming the engine reliably harvests edges when reversion is genuinely planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | plain-language ratio plot, ADF verdict, strategy vs coin, why 301-day half-life breaks the trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ADF + Engle-Granger + OU half-life, HAC t-stat on strategy vs random, cost sweep, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`gold_silver_ratio/`](gold_silver_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
