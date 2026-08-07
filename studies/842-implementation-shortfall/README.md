# Study 842 — Implementation Shortfall 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | A synthetic-only method demo: the gross edge is *planted*, not found on a real tape, so it can never earn `REAL` (which needs a robust *t* ≥ 2 on real data). On paper (0 cost) the book posts a dazzling **gross Sharpe 2.27** (NW *t* = **+7.65**) — but that is the *paper* number, capped at `NONE`. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Charge the friction of rotating ~35% of NAV a day and the 2.27 halves to **0.24** at a realistic cost (net *t* = **+0.79**, no longer significant) and goes **−1.78** when stressed. At higher turnover the identical paper alpha reaches net Sharpe **−15.54**. Nothing survives the trading. |
| **Does ignoring costs manufacture the edge?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Dropping the cost model turns a realistic +3%/yr nothing (or a −24%/yr stressed disaster) into a +31%/yr paper triumph; the gap is entirely the cost of trading, it scales with turnover, and the linear break-even (35 bp) hides it once market impact is in. The 20-seed control proves the gross edge is genuinely there to *be* eaten. |

> **In one sentence:** the same strategy that dazzles at zero cost — gross Sharpe 2.27 — is statistically dead at a realistic transaction cost and a money-loser when stressed, and it dies faster the more it trades, which is exactly why a backtest without a turnover-aware cost model is meaningless.

## What we tested

André Perold's *implementation shortfall* (1988): the frictionless "paper portfolio" and the real portfolio differ by the cost of actually trading into the positions, and that gap scales with turnover. We make it undeniable on a controlled tape — a moderate-turnover cross-sectional long-short with a **planted, genuine gross edge** — by evaluating the *identical* book across a cost ladder (0 / optimistic / realistic / stressed) with a **turnover-scaled market-impact term** (~ participation, super-linear in turnover). We show gross Sharpe vs net Sharpe down the ladder, the break-even cost, and the **turnover curve** where the same paper alpha goes from tradable to catastrophic as turnover rises. A 20-seed synthetic control proves the machinery fires on the plant and is silent on the null. *Synthetic-only by design (a real tape can't certify a clean planted edge), so it is capped at `NONE`.* **Dedup:** distinct from [30 House-Edge](../30-house-edge/) (a static retail markup on one instrument), [344 Backtest-Overfitting](../344-backtest-overfitting/) (a paper edge that never existed — manufactured by *search*, not killed by *costs*), and [619 BITO Roll-Drag](../619-bito-roll-drag/) (one product's structural roll cost, not a strategy's turnover-scaled impact).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the paper-vs-live gap in plain language: a Sharpe-2.27 "strategy" that a realistic cost quietly kills, and why trading more makes it worse |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: the dollar-neutral book, the linear + super-linear impact cost model, the cost ladder, the deceptive break-even, the turnover curve, HAC inference, and the 20-seed synthetic control |

The fingerprinted headline run (planted panel fp `79082a088002`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the whole machinery runs offline and deterministic on the synthetic world in [`cost_gap/data.py`](cost_gap/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`cost_gap/`](cost_gap/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
