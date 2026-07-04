# Study 633 — BTC Vol Targeting 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 30%-vol overlay tame BTC honestly? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the heart attacks, weak on the ride.* Max DD **−52.90% vs −83.40%**, robust across the whole 3×3 target/window grid, and certified as genuine **timing** by a 200-seed shuffled-vol placebo (**p = 0.010** — same-exposure random weights average −67% DD). But the return leg never clears the bar: HAC alpha **+7.18%/yr at *t* = 1.42** (placebo p = 0.105), grid alpha *t* = 0.44–1.48, Sharpe edge only +0.02…+0.11. Single-asset tape; BTC is itself the surviving coin — named. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Mechanically trivial: turnover **6.97× NAV/yr** (~0.5%/yr at 10 bps), unlimited capacity, any retail exchange — and every number survives **20 bps + 5% borrow** (Sharpe 1.045, DD −53.5%). But what survives certification is **risk control, not excess return**: you pay a realized **15.5 pp/yr CAGR give-up** for the smoother path, on one 11.8-year tape. A real shield, not a certified edge. |
| **"Same ride"?** | ![Busted](https://img.shields.io/badge/Same_ride%3F-Busted-8b949e?style=flat-square) | Terminal wealth **×43.9 vs ×152.6** — the overlay keeps **28.8%** of the buy-and-hold wealth (71% of the CAGR, +38.12% vs +53.63%/yr). *Half the heart attacks*: nearly literal (DD −53% vs −83%, vol 35.5% vs 66.5%). *Same ride*: no. (The growth gap's HAC *t* is −1.00 — not statistically certifiable on a tape this loud; the realized arithmetic is.) |

> **In one sentence:** scaling Bitcoin to a constant 30% vol target really does halve the heart attacks — the −83% drawdowns become −53%, the thermostat genuinely holds vol near 30% (median 31.0%), and a 200-seed placebo proves it's timing, not just holding less — but the "same ride" half of the pitch is busted (×44 vs ×153 terminal wealth, a 15.5 pp/yr realized give-up) and the Sharpe/alpha edge never certifies (*t* = 1.42), so it grades a Mixed signal in a Fragile, cheap-to-run vehicle.

## What we tested

The crypto-desk pitch that ports Moreira-Muir vol management to Bitcoin: hold
`w = min(1.5, 30% / RV30d)` of BTC-USD, rebalanced daily with **exactly one execution lag**
(the weight for day *t* uses returns through close *t−1*), cash at 0%. Tape: yfinance BTC-USD,
2014-09-17 → 2026-06-30 (4,305 daily bars, ann = 365, price-only = total-return). We race it
against buy & hold excess-vs-excess (same rf = 0 both legs): Sharpe, max DD, a **HAC t on the
daily log-wealth growth gap**, and a **HAC alpha regression** of strategy-on-B&H; a 3×3
**parameter grid** (targets 20/30/50% × windows 20/30/60d) checks robustness; a **200-seed
shuffled-vol placebo** splits *timing* from *mere exposure reduction*; a cost sweep charges
one-way bps × |Δw| daily plus a retail borrow spread on the levered fraction. A seeded synthetic
control (risk-priced null must earn nothing; planted leverage-effect world must light up — mean
*t* = 4.33, 100% of 20 seeds) proves the machinery. Distinct from
[210-crypto-trend](../210-crypto-trend/) (SMA **timing** — in or out) and
[591-vol-managed-portfolio](../591-vol-managed-portfolio/) (Moreira-Muir 1/RV on **equity
ETFs**): this is the continuous **vol-sizing overlay** on a single 66%-vol asset. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "targeting 30% vol" means, the thermostat chart, the heart-attack ledger (−53% vs −83%), and the honest price tag — you end with less than a third of the buy-and-hold wealth — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC growth-gap and alpha regressions, the 3×3 parameter grid, the 200-seed shuffled-signal placebo (alpha p = 0.105 · DD p = 0.010), the cost + borrow sweep, and the two-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`btc_vol_targeting/`](btc_vol_targeting/). The signal is trailing 30d realized vol
(one-day lag); the myth-check is the "same ride" claim. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
