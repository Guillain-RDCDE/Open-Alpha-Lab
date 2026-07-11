# Study 678 — Random-Walk-Index 🎲📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does RWI-high > 1 predict a better next-session return? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | No — if anything, backwards. SPY flag days pay **+2.41 bps** next session vs **+8.07 bps** on no-flag days (Welch *t* = **−1.64**); pooled across SPY/QQQ/IWM/DIA/GLD the wrong-signed gap **clears the bar** (Welch *t* = **−2.16**). A matched-count random-day placebo beats the flag's own mean in **96%** of draws. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The RWI-high long timer returns **+81.8% gross** over 21.5 years vs buy & hold's **+831.3%** (Sharpe 0.30 vs 0.64) and turns **net-negative at 10 bps** one-way costs. A block-shuffled, exposure-matched random-entry control **beats the real timer in 90% of draws by total return, 95% by Sharpe.** |
| **"Beats a coin?"** | ![Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square) | A same-exposure random-entry coin flip **outperforms** the RWI-timed book on both return and Sharpe most of the time — the "statistically non-random" trigger doesn't even clear the bar of picking days at random. |

> **In one sentence:** Poulos' Random Walk Index compares realized price displacement to what a
> pure random walk of the same ATR would produce, and claims RWI-high > 1 flags a real trend
> worth riding — but on SPY plus a four-name basket since 2005, flag days pay *less* than
> no-flag days (pooled Welch *t* = −2.16, wrong-signed), the resulting long timer captures a
> tenth of buy & hold's return before costs and goes negative after them, and a random-entry
> control with the same market exposure beats it outright.

## What we tested

The claim, in Poulos' own terms: a market that has moved farther over the last *n* sessions than
a random walk of the same Average True Range would be expected to move is "statistically
non-random" — RWI-high(n) = (High_t − Low_{t−n}) / (ATR_n(t) × √n), max over n = 2..6, and
**RWI-high > 1** is the mechanical "ride the trend" trigger. We flag every SPY session where the
indicator crosses that bar, measure the next session's return with a single documented execution
lag (flag known at close *t*, entered at that close, earns close(t)→close(t+1)), and run three
honest tests: flag-day vs no-flag-day return (Welch/Newey-West, a matched-count random-day
placebo), the resulting long timer vs buy & hold and vs a **block-shuffled, exposure-matched
random-entry control**, and cross-instrument pooling across SPY/QQQ/IWM/DIA/GLD so no single
tape decides it. A deterministic synthetic two-regime world proves the machinery: it stays quiet
on a null where trend and chop share the same drift, and lights up hard on a planted
trend-persistence edge. **Dedup:** siblings [108-adx-filter](../108-adx-filter/),
[484-vertical-horizontal-filter](../484-vertical-horizontal-filter/) and
[397-hurst-regime](../397-hurst-regime/) test the same "trend-strength gate" hypothesis with a
different formula each time (ADX, VHF, rolling Hurst) — none of them is the RWI, and all three
land the same place this one does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Random Walk Index is trying to measure, why "moved farther than random" sounds like a free lunch, and why the actual trade loses to a coin flip — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC flag-day split, the matched-count placebo, the block-shuffled exposure-matched control, cross-instrument pooling, cost sweeps, and the synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`random_walk_index/`](random_walk_index/). RWI-high = max over n = 2..6 of
(High_t − Low_{t−n}) / (ATR_n(t) × √n), simple (non-Wilder) ATR. No survivorship — SPY/QQQ/IWM/DIA
are index ETFs, GLD tracks a physical commodity, none of it a stock-picking panel. **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
