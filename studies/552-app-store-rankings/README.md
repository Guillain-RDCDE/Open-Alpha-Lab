# Study 552 — App-Store-Rankings 📱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**When a public company's app climbs the App Store charts, does the stock climb with it?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does app-ranking momentum nowcast forward returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The mechanism (app rank as a real-time demand nowcast) is **literature-plausible** and our engine is a **faithful detector** — the seed-robust IC-*t* ramps smoothly past the bar when the effect is planted (0.15 → **+2.62**) and stays flat at the null (**+0.12**). **But there is no free, point-in-time, survivorship-clean App Store ranking panel** a retail stack can reach, so **no real tape was tested** and there is no robust *t* ≥ 2 on real data — `WEAK` by rule, never `REAL`. Even the *planted* signal falls below the bar as the rank read gets noisy (signal_noise 4.0 → *t* +0.92), which is the real-world regime. |
| **Tradability** — does the long-short pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A monthly-rebalanced top-minus-bottom-tercile book whose **short leg is the sinking-app, small/hard-to-borrow tail**. In the honest (null) world the spread is negative before costs (gross **−0.5%/mo**, net **−1.0%/mo** after 10 bps/leg × 4 crossings + 100 bps/yr borrow); only a *planted* effect makes it pay. Nothing to harvest on any signal you could actually buy. |

> **In one sentence:** app-ranking momentum is a plausible, engine-validated demand nowcast — but there is **no free real tape** to prove it (so it's capped at `WEAK`, not `REAL`), the same effect fails to certify once the rank read is realistically noisy, and the long-short bleeds after a monthly rebalance and a borrow on the hard-to-short sinking-app leg.

## What we tested

The alt-data claim: a consumer-tech company's **App Store download rank** is a real-time proxy for
demand, so when its app **climbs the charts** the stock should climb with it — ranking momentum as a
fundamental nowcast, the *broad cross-sectional* cousin of the single-name
[294 Coinbase-Rank](../../294-coinbase-rank/) omen. Because **no free, point-in-time,
survivorship-clean ranking panel exists** (Apple has no historical rank API; vendor history is
licensed and modelled), this is a **synthetic-only** study: a deterministic panel (seed = 552) with
one knob planting the ranking→return effect. We build the standard alt-data toolkit — a
cross-sectional **information coefficient** (Spearman IC of ranking-improvement vs forward return)
and its *t*-stat, a long-short tercile spread reported **gross AND net** of costs + a punitive short
borrow, a **label-shuffle placebo** null, a multi-window robustness sweep, and a **seed-robust
synthetic positive control** (≥ 20 seeds) that proves the engine catches a planted effect and stays
flat at the null. The honest headline is the **null world** (no signal) — correctly flat — plus the
data-availability wall named on the SIGNAL axis. One execution lag; the short leg pays borrow.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "app rank as a demand nowcast" means, why an app isn't a ticker, why the honest run is flat, and why you can't trade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Spearman IC and its *t*, the placebo null, the four-window sign wander, costs + borrow, the seed-robust positive control, and the signal-noise sweep that sinks the effect |

The fingerprinted, deterministic headline run (null history fp `fc780c9f143d`, planted fp
`5b92f4d9fdb5`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
lives in [`app_store_rankings/`](app_store_rankings/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`app_store_rankings/`](app_store_rankings/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
