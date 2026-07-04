# Study 618 — GBTC Premium Cycle 🎁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — one wrapper, three regimes, all mechanical? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | All three regimes are on the tape: premium era **+36.3 %** (HAC *t* = **+9.0**), discount era **−24.5 %** (HAC *t* = **−7.4**, trough **−48.8 %**), ETF era **−0.03 %** — and the dated 2023 convergence clears the bar on its honest ex-ante version (enter one day after BlackRock's filing): **HAC *t* = +2.69**, +43.5 log-pts, with **3.5–3.7 sd** one-day moves on the two documented catalysts. Full-calendar-2023 (the hindsight window) reads *t* = 1.92 — said out loud. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The 2023 convergence was genuinely deployable — any brokerage, billions of capacity, **+49.6 % net** of costs and borrow in 7 months. But it was a **one-shot, dated trade that no longer exists**: since 2024-01-11 the in-kind arb pins GBTC to NAV (mean \|premium\| **0.6 %**). Real, harvested, extinct. |
| **Who could harvest each regime?** | ![Mixed](https://img.shields.io/badge/Who_could_harvest%3F-Mixed-8b949e?style=flat-square) | Premium era: **accredited only** — create at NAV, dump at the premium after the lockup (**+26.5 log-pts**/cohort, 98 % hit rate)… until the flip turned the same trade into **−17 to −23** (the 3AC/BlockFi widow-maker; early-vs-late Welch *t* = 17.8). The public could only *pay* the premium. Discount era: **anyone**. ETF era: **no one** — the arbitrage became the product. |

> **In one sentence:** GBTC really did trade ~40 % above its bitcoin while accredited creators
> dumped into locked-out retail demand, ~49 % below once the no-redemption clause trapped the
> coins, and snapped to NAV the day the ETF arb switched on — three mechanical regimes we
> reconstruct to basis-point precision (ETF-era residual −0.03 %), with the one publicly
> harvestable leg (the 2023 convergence, HAC *t* = +2.69, +49.6 % net) now permanently closed.

## What we tested

We rebuild GBTC's **premium/discount to held bitcoin** over its whole listed life (2015-05-11 →
2026-06-30, yfinance daily GBTC + BTC-USD) against a **modeled bitcoin-per-share** path built
from public trust mechanics — 0.1 BTC/share at the 2013 inception, the 2 %/yr fee accrued in
BTC, the 91-for-1 split, the 1.5 %/yr ETF fee and the ×0.90 Mini-Trust spin-off — with the
ETF era as the built-in calibration check (the arb forces premium ≈ 0; ours reads −0.03 %).
The Signal axis measures the three documented regimes (HAC *t* on levels, Welch *t* across
them) and puts the decisive inference on the **2023 convergence drift**: enter at the close one
trading day after BlackRock's 2023-06-15 spot-ETF filing, exit at the conversion close, HAC *t*
on the long-GBTC/short-BTC daily return. Tradability charges 10–50 bps legs plus 5 %/yr short
borrow. The third axis prices the premium-era **accredited create-and-dump** through its
Rule-144 lockup, cohort by cohort, including the 2020-21 blow-up. A planted three-regime
synthetic world proves the machinery (levels recovered; drift *t* lights up only when planted).
As-of **2026-06-30**, fingerprint `5ffd3440484c`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how one fund traded 40 % rich then 49 % cheap for the *same* coins, who was minting money off it (and who blew up), and why an ETF stamp deleted the whole game overnight — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the BTC-per-share reconstruction + its basis-point calibration, regime HAC/Welch stats, the ex-ante convergence trade with HAC *t* and costs, catalyst event-day z's, the lockup-arb cohort arithmetic, and the planted-regime synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gbtc_premium_cycle/`](gbtc_premium_cycle/). Sibling studies: [324-bitcoin-treasury](../324-bitcoin-treasury/) is **MSTR** — bitcoin wrapped in an operating company; this is the **fund wrapper's arb lifecycle** (creation gates → no redemption → in-kind arb). BTC itself: [70-digital-gold](../70-digital-gold/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
