# Study 582 — ETH-Gas-Fees ⛽

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do soaring Ethereum gas fees — the price of using the chain — signal a crypto top or genuine demand?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do gas spikes predict low forward ETH returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **Synthetic-only** — a daily, survivorship-free gas-fee tape needs an archive node / keyed API, out of reach for a no-key stack, so there is **no REAL tape** to clear a *t* ≥ 2. On the null world the spike−calm gap is **+0.07%** (*t* +0.17), slope-*t* **−0.04**, placebo *p* **0.856** — flat, as it must read. |
| **Tradability** — does the contrarian short pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A short-ETH-on-euphoric-congestion overlay *loses even gross* (**−16.2%**/yr, HAC *t* −1.00) and worse net (**−21.2%**/yr after 10 bps + a 1000 bps/yr crypto borrow). Shorting the coin's drift with a heavy borrow is a bleed. |
| **"Top signal on-chain?"** | ![Unproven](https://img.shields.io/badge/Unproven-8b949e?style=flat-square) | Not busted, not confirmed — **untested on real data**. The mechanism is coherent and the engine is proven faithful (flat at the null, slope-*t* past −2 when a top signal is planted, seed-robust over 25 seeds), but no reachable gas tape decides it. EIP-1559 + the L2 migration also change what "gas" even means. |

> **In one sentence:** "gas spikes mark the top" is a coherent, testable story with **no reachable real tape** on a retail stack — so we build a deterministic gas/price world, prove the engine would catch a planted top signal (past *t* = −2, seed-robust), show that at the null it correctly reads *nothing* and the contrarian short bleeds even gross, and leave the myth honestly **unproven** until someone brings an archive-node gas series.

## What we tested

The folk claim: when Ethereum **gas fees** spike, the chain is congested with mints, swaps and
leverage — euphoric activity that (contrarian read) marks a **top**, so *forward* ETH returns
should be LOW; the bullish counter-read is that high gas is *genuine demand*. Because a clean daily
gas-fee series is unreachable without an archive node or keyed API, this is a **synthetic-only**
study: a deterministic coupled (gas, ETH price) world where congestion follows euphoria, with a
single knob (`top_signal_beta`) that plants the contrarian effect. We build a trailing MAD-scaled
**gas spike**, a short-on-top-decile **contrarian overlay**, a two-sample gap and a HAC
forward-return slope (whose *sign* is the claim), a **label-shuffle placebo** null, gross-and-net
costs with a punitive crypto **short borrow**, a **five-horizon** robustness sweep, and a
**seed-robust (25-seed) synthetic positive control**. The ETH-USD price tape (yfinance) is real —
used only to calibrate the return distribution and anchor the data-availability limit, **not** to
compute a gas→return headline (there is no real gas series). *Distinct from
[325 Crypto-Fear-Greed](../325-crypto-fear-greed/) (a survey sentiment gauge) and
[292 Bitcoin-Hashrate](../292-bitcoin-hashrate/) / [295 Stablecoin-Supply](../295-stablecoin-supply/)
(other on-chain metrics) — this is the **blockspace-congestion** proxy.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what gas fees are, why a spike *might* mean a top, why we can't test it on real data, and what a fair synthetic test shows |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the gas-spike construction, the spike−calm gap with a two-sample *t*, the HAC slope, the placebo null, the horizon sweep, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted run (null world fp `0e87032ad333`; ETH anchor fp `d4b0cdbf2e5e`, 3101 real daily
returns) is in [docs/results.md](docs/results.md); the offline machinery runs on the deterministic
synthetic world in [`eth_gas_fees/data.py`](eth_gas_fees/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`eth_gas_fees/`](eth_gas_fees/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
