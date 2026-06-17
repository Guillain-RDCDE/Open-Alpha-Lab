# Study 301 — Triple-RSI

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | SPY gross **+132.8 bps/trade**, HAC *t* = **+5.07**; survives a look-ahead-free next-open fill (*t* = +4.22) and the post-2010 half (*t* = +3.11); beats a random-direction control by **+84.9 bps**. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Costs barely dent it (still *t* = +4.69 at 10 bps) — but only **3.5 trades/yr**, ~**7% of the time in the market**, so it compounds to ~**4.7%/yr** vs SPY buy-and-hold's **10.8%/yr**. A capital-starved overlay, not a system. |
| **A 90% win-rate money machine?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The **88%** win-rate is real — but on a pure-coin tape this same "take the bounce" exit *already* prints 62% wins with a negative mean. Win-rate ≠ expectancy; skew is **−1.6**. |

> **In one sentence:** the Triple-RSI oversold bounce is a genuinely real, cost-robust signal and the ~90% win-rate is no lie — but at three-and-a-half trades a year it earns half of buy-and-hold while sitting in cash, and the headline number is the oldest illusion in the book.

## What we tested

QuantifiedStrategies.com (and a chorus of vendor backtests) market the **"Triple RSI"** system as a **90% win-rate** edge on SPY: buy at the close when (1) the 5-day RSI is below 30, (2) RSI fell for the third day running, (3) RSI three days ago was below 60, and (4) the close is above the 200-day average — then sell at the close when RSI(5) crosses back above 50. We take that literally on SPY daily bars since 1993 (plus QQQ, IWM, DIA for breadth), pin it against a **random-direction control** on the same entry bars, re-run it with a conservative next-open fill to rule out same-bar look-ahead, sweep costs, and split pre/post-2010. A deterministic synthetic tape with tunable mean-reversion is the positive control — and doubles as the cleanest demonstration of why a high win-rate proves nothing.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, why the bounce is real, and the win-rate trap shown on a literal coin |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, random control, next-open robustness, cost sweep, capacity, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`triple_rsi/`](triple_rsi/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
