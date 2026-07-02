# Study 585 — Perp-Funding-Rate 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the funding-rate contrarian effect real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The desk's `REAL` bar is a robust *t* ≥ 2 **on a real tape** — and the real tape doesn't exist for a no-key stack: **funding history is paid/rate-limited exchange data** (Binance/Bybit `fapi`, Coinglass, Amberdata). Free data (yfinance) reaches BTC/ETH *prices* but **not** funding, so there is no real number to certify. The **synthetic control** proves the engine *would* catch it (spread **+3.01%**, slope *t* **−9.44**, placebo *p* **0.0005**, sign stable across 1→9-period horizons, flat at the null), but synthetic-only caps at `NONE`/`WEAK`. |
| **Tradability** — would the flush pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even if real, it's the **highest-turnover trade on the desk** — an 8-hourly rebalance — and the contrarian **short** leg literally *pays* the positive funding it keys on. On the synthetic tape gross **+3.01%** → net **+2.47%** (6 bps/leg + 30 bps short-side funding drag) *per rebalance* — but that's a *planted* effect, not a market fact. |

> **In one sentence:** "extreme funding = crowded longs due for a flush" is a plausible, well-liked crypto-desk story, and our synthetic control shows that *if* it were real this engine would catch it cleanly — but the funding-rate tape that would make it a **real** test is locked behind paid exchange APIs, so the reproducible core is synthetic by necessity and the study is capped below `REAL` (`NONE` × `MIRAGE`) until someone pipes in a live funding feed.

## What we tested

The **perp-funding contrarian claim**: a crypto perpetual swap is tethered to spot by a periodic
**funding rate** paid between longs and shorts; when funding runs **hot** (large positive — longs
paying shorts) the crowd is long-and-levered and a mean-reversion flush is due, so extreme funding
is a **contrarian short** signal for the forward return (and extreme *negative* funding a contrarian
long). Because a no-key retail stack reaches BTC/ETH **prices** but *not* the historical
**funding-rate tape** (paid/rate-limited exchange data), the reproducible core is a deterministic,
seeded **synthetic** funding + forward-return panel with one planted-effect knob (`contrarian_beta`).
We standardize funding (trailing z), sort into hot/cold quintile tails, and run a two-sample *t* on
the cold-minus-hot spread, a **label-shuffle placebo** null, a period-level slope (whose *sign* is
the claim), a **horizon sweep**, costs + the short-side funding drag, and a **seed-robust synthetic
positive control** (25 seeds) that plants the effect and proves the engine catches it while staying
flat at the null. *Distinct from the desk's price/calendar/sentiment crypto studies
([133](../133-crypto-seasonality/), [175](../175-crypto-weekend/), [210](../210-crypto-trend/),
[251](../251-crypto-reversal/), [325](../325-crypto-fear-greed/)) — this is the one crypto signal
whose entire input is not free; the synthetic core matches the desk's other synthetic-only studies
([273](../273-lego-returns/), [275](../275-whisky-cask/), [276](../276-sneaker-resale/)).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what funding is, why hot funding might mean a flush is coming, what the synthetic world shows, and why we *can't* test it for real |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the trailing-z signal, the quintile sort with a two-sample *t*, the placebo null, the period-level slope, the horizon sweep, costs + funding drag, and the seed-robust synthetic control |

The fingerprinted synthetic headline run (seed 585, planted `contrarian_beta = -0.012`, panel fp
`8121108453d2`) is in [docs/results.md](docs/results.md); the free BTC/ETH price tape (fp
`db49481bb5f1`) is cached only to make the missing-funding-data limitation concrete. The offline
machinery lives in [`perp_funding_rate/`](perp_funding_rate/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`perp_funding_rate/`](perp_funding_rate/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
