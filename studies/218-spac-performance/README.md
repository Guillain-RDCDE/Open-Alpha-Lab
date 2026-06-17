# Study 218 — SPAC-Performance

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-NONE-c0392b?style=flat-square) | Jensen alpha **-30.69%/yr** on SPAK (1.92 yrs), HAC *t* = **-1.925** — short window limits power; raw excess HAC *t* = **-2.000** confirms direction. De-SPAC basket: alpha **-31.52%/yr**, *t* = -1.926 over 4.56 yrs. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-MIRAGE-c0392b?style=flat-square) | SPAK CAGR **-26.14%/yr** vs SPY **+10.26%/yr**; Sharpe **-0.93** vs **+0.63**; max DD **-62.6%** vs **-23.0%**. Basket: -86.3% max DD; SPAK delisted 2022 as AUM collapsed. |
| **Were SPACs a clever shortcut to public markets or a structurally rigged loss?** | ![Busted](https://img.shields.io/badge/SPACs%3A_shortcut_or_rigged%3F-Busted-8b949e?style=flat-square) | The ~20% sponsor "promote" plus warrants structurally dilutes public investors by ~12% at merger before any operational miss. SPAK underperformed SPY by **36 percentage points per year** over its full life. |

> **In one sentence:** SPACs transferred ~$10 of every $100 invested from public shareholders to sponsors via structural dilution, then compounded the loss through overvalued merger targets — the de-SPAC universe lost over 60% peak-to-trough while the S&P 500 gained, a structural rigging, not a market cycle.

## What we tested

The SPAC boom (2020-2021) produced hundreds of blank-check vehicles claiming to offer
a democratised, faster, and less dilutive path to public markets. We test two tapes:

1. **SPAK** (Defiance Next Gen SPAC Derived ETF, ER ~0.45%) — the purpose-built de-SPAC
   index ETF, active from its launch (2020-10-01) to delisting (2022-09-01), 483 trading
   days. This gives a clean, diversified, index-level view of the SPAC universe.

2. **De-SPAC basket** (9 surviving names: LCID, RIVN, OPEN, PSFE, CLOV, SKLZ, DKNG,
   SPCE, QS; equal-weight from 2021-11-11 through 2026-06-12, 1,149 days). This extends
   the window but carries **survivorship bias upward** — delisted names (NKLA, OTRK,
   HYLIION…) are absent, meaning true cohort losses are even larger.

Both are compared to SPY (S&P 500, ER ~0.09%) using CAPM Jensen alpha with Newey-West
HAC t-statistics. The structural-dilution mechanics (sponsor promote, warrants, redemption
drain) are documented and already baked into the total-return price series.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what SPACs promised, what the promote actually means in dollar terms, the stunning SPAK vs S&P 500 track record in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Jensen alpha, HAC t-stat on both tapes, survivorship-bias quantification, synthetic positive control confirming detection works |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`spac_performance/`](spac_performance/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
