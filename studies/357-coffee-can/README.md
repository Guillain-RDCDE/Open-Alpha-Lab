# Study 357 — the Coffee-Can portfolio ☕

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a never-trade quality basket beat the index? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | **No on return** — the can compounds at **9.49%/yr** vs SPY's **11.11%**; the monthly-excess paired HAC *t* = **−0.83** (n = 256), wrong sign and insignificant. **Yes on risk** — half the drawdown (**−31%** vs **−51%**) and lower vol, so Sharpe **ties** (0.79 vs 0.78). It de-risks the index; it doesn't beat it. |
| **Tradability** — is there an edge to harvest? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The one honest tailwind (zero turnover → no spread/commission/tax) is **~1bp/yr** at a sane rebalance pace, and the can even **trails its own rebalanced twin** (9.49% vs 10.28%). Real but immaterial against the index. |
| **Survivorship?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | On a synthetic market with **no skill** (every firm shares an 8%/yr drift), choosing the basket by hindsight fabricates **+8.2 pp/yr** over an honest ex-ante pick (**+12.4** vs the death-inclusive universe). The legend's "magic" is mostly choosing winners after the fact. |

> **In one sentence:** the Coffee Can — buy great names and never sell — is a fine *low-stress, low-cost, defensive* way to **own** the market (calmer ride, half the crash), but it does **not beat** it: its excess return is negative and insignificant, the cost edge is a rounding error at a sane pace, and the storied out-performance is mostly **survivorship** — you're picking the winners with hindsight.

## What we tested

Robert Kirby's 1984 *Coffee Can Portfolio* parable says: equal-weight a basket of quality companies, drop the certificates in a can, and **never trade for a decade** — you'll beat the fussing active managers because you won't churn, won't pay costs/taxes, and won't sell your winners too early. We take **10 quality large-caps** (KO, JNJ, PG, PEP, WMT, MMM, MO, XOM, IBM, GE) and **21 years** of real monthly **total-return** prices (yfinance, splits + dividends), and run a zero-turnover buy-and-hold against (a) a yearly-rebalanced twin charged realistic costs on NAV and (b) the index (SPY). The Signal test is a paired HAC *t* of the can's monthly excess over the index. Then the honesty layer: on a **synthetic market engineered to have no edge at all**, we build the can by **hindsight** (famous terminal winners) vs **honestly** (picked ex-ante) to measure the survivorship inflation directly. (Same risk-not-return signature as the lazy-portfolio family — see [Study 144](../../144-permanent-portfolio/) — and the same dead-names bias as [Study 151](../../151-stocks-for-long-run/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the can rides smoother but finishes behind, where the "magic" really comes from, and how hindsight invents an edge from nothing — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the paired HAC *t* of excess return, the risk decomposition (vol/MDD/Sharpe), the cost-on-NAV sweep, and a deterministic survivorship control on a no-skill panel |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`coffee_can/`](coffee_can/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
