# Study 962 — Do It Yourself 🧰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a **contemporaneous** name list the blended annualised gap is **−0.76%** (HAC *t* = **−0.58**, bootstrap CI [−3.21%, +1.69%]); no sector clears |*t*| = 2 and no sign survives the era cut. Every |*t*| ≥ 2 result in the basket's favour comes from the **hindsight** basket (7 of its 9 cells; blended +6.08%/yr, *t* = +6.10). Against an equal-weight 2026 control that difference splits into a **+5.81%/yr look-ahead** in the membership list (positive in all three sectors) and **+1.02%/yr** of plain cap- versus equal-weighting (positive only in XLK) — so the look-ahead is +5.81%, not the raw +6.84%. Survivorship named twice: the hindsight list *is* a survivor list, and the 2011 control is survivor-conditional throughout (in energy explicitly — two acquired names have no tape) in the DIY basket's favour. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The fee saved is **0.08%/yr**; the tracking error accepted to save it is **5.13%/yr** blended and 9–11%/yr per sector — **64× to 134×** the prize — plus a worst rolling 12-month shortfall of −12.3% (−25.7% in energy) and an energy drawdown 10.5 pp *deeper* than the fund's. The standard error on the gap is 16× the fee: this test could not resolve it in 16,500 years of tape. |

> **In one sentence:** Holding a sector fund's top ten names instead of the fund does not buy you the 0.08% expense ratio — it buys you an undiagnosed **active bet** with 5–11% annualised tracking error in both directions, and the only version that "works" is the one built from the holdings list published fifteen years after the back-test starts.

## What we tested

Replicate **XLK**, **XLE** and **XLF** with their own top **N = 3 / 5 / 10** holdings,
monthly rebalanced, one-day execution lag, 5 bps one-way × NAV, and race each basket
against the fund it replaces on daily total-return closes 2011-01-03 → 2026-06-30. Three
baskets isolate the look-ahead: the fund's **published cap weights as of 2026-06-30** (the
hindsight basket), its **January-2011 top-10 held equal weight, fixed from day one** (the
control), and **today's names at equal weight** — the third one exists because the first
two differ in *two* things at once, and only holding the weighting scheme fixed separates
hindsight from cap-weighting. HAC *t* on the daily gap, block-bootstrap CI, excess-of-cash
Sharpe race, era cut, cost and rebalance-frequency sweeps. Both holdings lists are
**hardcoded PROXIES**, the 2011 list is **survivor-conditional**, and tax is modelled
nowhere — all labelled in [docs/results.md](docs/results.md).
**Dedup:** distinct from **920-total-cost-of-ownership**
(fee-vs-friction between two *wrappers*), **913-tracking-difference-persistence** (two funds
on the *same* index — a difference two orders of magnitude smaller),
**177-megacap-concentration** (do big stocks beat *the market* — a return question, not a
tracking one), **28-carousel** / **225-sector-rotation** (choosing *between* sectors; we
never rotate), **890-sector-risk-parity** / **903-sector-neutral-lowvol** (re-weighting
*across* sectors, not inside one), and **870-industry-leader-lead-lag** (the big name as a
*signal*, not as the portfolio).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the forum argument sounds right, the hindsight trap in one table, what 5% tracking error feels like year by year, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the depth × weighting sweep, HAC *t* on the gap, bootstrap CIs, the power arithmetic, era cut, cost and frequency sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`diy_sector/`](diy_sector/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
