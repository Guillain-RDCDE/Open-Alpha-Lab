# Study 941 — Short Both Legs ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Short the 3x long *and* the 3x short of the same index — is the decay harvest free money?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Real](https://img.shields.io/badge/Real-27ae60?style=flat-square) | **Before any borrow fee**, shorting TQQQ and SQQQ in equal dollars with a daily reset earns **+2.06%/yr excess-of-cash** (HAC *t* = **+8.82**, bootstrap mean CI [+1.67%, +2.64%], residual beta to QQQ **+0.003**, alpha +2.00%/yr), positive in all 17 calendar years (2026 = H1 stub), in both eras, and surviving the loss of its 50 best days (+0.95%/yr) or of 2020 (+1.77%/yr) as magnitude checks. But it is **not the decay**: it is the two funds' residual shortfall against a costless ±3x replication, and the synthetic control pays nothing when an identical 14%/29% decay carries no fee. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-e67e22?style=flat-square) | The whole edge sits inside one unobservable: **breakeven blended borrow is 2.06%/yr**, and this pair is cheap TQQQ (2.79%/yr of shortfall, 63% of the harvest) subsidising crowded SQQQ (**1.61%/yr**, the thin leg). At the blended breakeven **eleven of the seventeen years lose money**. Without a rebate on the short proceeds it earns +0.75%/yr overall and **+0.10%/yr (*t* = +0.22) since 2018** — retail terms make it a Mirage. Rebalancing less triples the headline but the extra is residual beta (alpha *t* = +0.68), and an unrebalanced book is wiped out on 2013-08-01 at 51.7x gross. Pair selection runs *against* the trade: this is the easiest levered pair in the world to borrow. |

> **In one sentence:** The double-short *does* pay a real, market-neutral ~2%/yr before borrow — but what it collects is the two funds' own cost load, not the famous volatility decay, so the harvest is precisely the size of the borrow fee that exists to price it away.

> **One honesty note, up front.** On the daily reset the book's excess return is *algebraically identical* to `cash − 0.5 × (TQQQ + SQQQ)`, the "implied fund load" quoted in the docs. The two agreeing is arithmetic restating itself, **not** two independent measurements — and the +2.20%/yr is a **residual** (expenses + financing spread + the funds' internal trading + tracking), none of whose components this study measures separately.

## What we tested

Per $1 of equity, short **$0.50 TQQQ** (3x long Nasdaq-100) and **$0.50 SQQQ** (−3x
inverse), reset to equal dollars **daily / weekly / monthly / never**, proceeds and equity
in **BIL**, one execution lag (reset decided at the close of day t, effective day t+1),
2 bps one-way cost, everything **excess-of-cash** on total-return closes over
2010-02-11 → 2026-06-30. Borrow (0–15%/yr) and the short rebate are labelled
**ASSUMPTIONS** and swept; the breakeven borrow *is* the headline. HAC *t*, block-bootstrap
CIs, residual beta to QQQ, a per-leg decomposition of the harvest, an era cut, an
outlier-robustness cut, and a synthetic control that separates cost load from decay.
The universe is **n = 1 and hindsight-chosen** (the one 3x pair with a full history) —
conservative for the signal, anti-conservative for the borrow; both directions are named.
**Dedup:** distinct from **593-hfea-leveraged-6040** / **594-leverage-rotation-200sma**
(levered ETFs held *long* as an allocation), **943-leverage-reset-frequency** (the *fund's*
reset schedule, not the arbitrageur's), **154-leverage-anomaly** (cross-sectional), and from
**375-vxx-roll-decay** / **661-uso-roll-decay**, where shorting a decaying product works
because that decay is a real roll yield rather than a compounding artefact.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the free-money chart is misleading, what you actually collect, why borrow decides everything, the day the unrebalanced book died |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the arithmetic-vs-geometric decomposition, HAC *t* and bootstrap CIs, residual beta, borrow and rebate sweeps, the schedule race, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`short_pair/`](short_pair/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
