# Study 100 — Melting-Ice 🪫

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The decay *mechanic* is exact: the constant-leverage daily identity reproduces real TQQQ/UPRO at daily-return **corr 0.999** (RMS tracking error <20 bps, final NAV within ~2% over 16 years). Variance drag is real maths — `(k(k−1)/2)·σ²`. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Hold 3x forever" earned **+18–24 pts/yr** of CAGR — but at a **lower Sharpe** than the underlying (more risk per unit return) and a **−77% to −82%** drawdown. It's a regime-conditional leveraged-beta bet, not durable alpha. |
| **Always decays to zero?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | In **both** real 2010s tapes the realised 3x NAV ended **far above** "3x the period return" (TQQQ x369 vs x55; UPRO x121 vs x31). Decay is **path-dependent**: trends help, chop hurts. |

> **In one sentence:** "volatility decay guarantees a 3x ETF bleeds to zero" is **busted** — the drag is real maths but it's a *race* against compounding, and in the trending 2010s compounding won by a mile (TQQQ beat naive-3x ~7-fold); the genuine catch isn't a decay law, it's the **ruinous −80% drawdown** and lower-than-underlying Sharpe.

## What we tested

The most-repeated warning on leveraged ETFs, stated at full strength: *"3x funds like
**TQQQ** and **UPRO** are toxic — **volatility decay** guarantees they bleed to zero over
time; **never** hold them more than a day."* We take the mechanic literally — simulate the
exact constant-leverage daily rebalancing identity (`L_t = L_{t−1}·(1 + 3·r_t − fee)`,
all-in fee ≈ 5%/yr) from the underlying's daily total return — **validate it against the
real funds** (corr 0.999), then compare the realised 3x NAV to the naive "3x the period
return" a believer expects and decompose the gap into the **variance-drag** term
(`(k(k−1)/2)·σ²`, negative) and the **compounding-in-a-trend** term (positive), citing
Cheng & Madhavan (2009) and Avellaneda & Zhang (2010). We compute the break-even
volatility at which drag overtakes drift. A deterministic two-sided synthetic control (a
high-vol **flat** tape where 3x decays, a low-vol **uptrend** where 3x compounds ahead)
proves the harness reports the *sign* of the gap correctly.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the doom claim, the real TQQQ/UPRO curves, why daily resetting helps in a trend and hurts in chop, and the −80% drawdown that's the *real* danger |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the exact rebalancing identity, the drag-vs-drift decomposition, break-even volatility, the by-regime path-dependence, Sharpe-vs-underlying |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`melting_ice/`](melting_ice/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
