# Study 583 — DeFi-TVL-Momentum 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**When money floods into DeFi protocols (rising total value locked), do their tokens keep pumping?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does TVL flow predict token returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | **No real tape exists.** A faithful test needs a survivorship-free, point-in-time panel of per-protocol TVL joined to token returns *with the tokens that went to zero still in it* — unreachable on a free, no-key stack. On a **planted** synthetic world the engine banks the effect cleanly (long-short *t* **+10.9**, placebo *p* 0.0005, slope-*t* +11.3) and stays flat at the null (*t* +0.33). But with no robust real tape, a synthetic-only study is capped at `WEAK`. |
| **Tradability** — does the flow sort pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Monthly rebalance × two legs at crypto-scale spreads, plus an **800 bps/yr borrow on a short leg you could never actually borrow** — the outflow leg is the rug/exploit tail that gaps to zero. Even in the planted world net trails gross (+10.5%/mo vs +11.8%/mo); on a real tape the short is uninvestable and the capacity is a rounding error. |

> **In one sentence:** the "TVL flooding in → token pumps" story is testable machinery — and our engine catches a planted flow effect at *t* +10.9 while staying flat at the null — but there is **no survivorship-free, point-in-time real tape** a retail stack can reach, so the signal is `WEAK` (never certifiable `REAL`) and the trade is a `MIRAGE`: the short leg is the tokens that went to zero, which no one lets you borrow.

## What we tested

The **on-chain folklore**: total value locked (TVL) is *"smart money"* flowing in, a leading
public signal, so a cross-sectional **TVL-momentum** sort — long the fastest-inflow protocols,
short the outflows — should print a positive forward-return spread. We build a deterministic,
seeded synthetic monthly panel (60 protocols × 47 months) whose single knob `tvl_alpha` plants the
effect (`>0`) or turns it off (`=0`), then run the honest machinery: the tercile long-short with a
one-sample *t*, a **label-shuffle placebo** null, a month-clustered protocol-level slope, a
formation/holding × basket-fraction robustness sweep, crypto-scale costs + a punitive short borrow,
and a **seed-robust (25-seed) synthetic positive control** proving the engine catches the planted
effect and stays flat at the null. **There is no real tape** — a survivorship-free, point-in-time
per-protocol TVL × token-return panel is not reachable on a free, no-key stack (like the desk's
[273 Lego-Returns](../../273-lego-returns/) / [275 Whisky-Cask](../../275-whisky-cask/) /
[276 Sneaker-Resale](../../276-sneaker-resale/) synthetic-only studies) — so the data-availability
wall is named on the Signal axis and the stamp is capped at `WEAK`. *Distinct from the desk's
price-derived crypto studies ([133 Crypto-Seasonality](../../133-crypto-seasonality/),
[210 Crypto-Trend](../../210-crypto-trend/)) and the TradFi flow cousin
[561 ETF-Flow-Momentum](../../561-etf-flow-momentum/): this is the **on-chain fundamental flow**
(TVL) as the signal.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what TVL is, why "money flooding in" sounds like a signal, and why we can't get an honest tape to test it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the tercile long-short with a one-sample *t*, the placebo null, the month-clustered slope, the robustness sweep, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted synthetic run (planted panel fp `0b6d95a52429`, null fp `0a3c150d737e`, as-of
2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery lives in
[`defi_tvl_momentum/`](defi_tvl_momentum/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`defi_tvl_momentum/`](defi_tvl_momentum/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
