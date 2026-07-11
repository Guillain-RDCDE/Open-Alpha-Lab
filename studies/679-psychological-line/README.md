# Study 679 — Psychological-Line

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Oversold forward return **+159 bps/5d**, Welch *t* = **+1.32**; overbought **+66 bps/5d**, Welch *t* = **+1.04** — neither clears *t* ≥ 2, and the overbought leg's sign is *wrong* for the claim (more positive, not negative). Every one of six instruments individually |*t*| < 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Zone-trigger timer nets **−31.95 bps/trade** at 5 bps cost (HAC *t* = −1.49), **−41.95 bps** at 10 bps (*t* = −1.95). No cost level makes it money. |
| **Beats a coin?** | ![Mixed](https://img.shields.io/badge/Beats_a_coin%3F-Mixed-8b949e?style=flat-square) | A random-direction control on the identical entry bars earns **+18.57 bps/trade** where the PSY timer loses **−31.95** — a 50.5 bps gap favoring the coin — but the gap's own Welch *t* = **−1.55** doesn't clear certification. |

> **In one sentence:** the 12-day Psychological Line's textbook 75/25 overbought/oversold read carries no certifiable forward-return signal on SPY + a five-name basket over 23 years — the oversold leg is directionally right but too thin to certify, the overbought leg runs the *wrong* way, and actually trading the zone crosses loses money net of costs while a same-bar coin flip does not.

## What we tested

PSY(*N*) = 100 x (up-closes in the last *N* days) / *N* — a Japanese charting-era crowd-sentiment
gauge (Nison, *Beyond Candlesticks*), read the textbook way: **PSY > 75 → overbought, sell**;
**PSY < 25 → oversold, buy**, *N* = 12. We bucket every trading day on SPY, QQQ, IWM, AAPL, TSLA
and NVDA (2003→2026) by its PSY reading and compare the forward 5-day return of the zone-entry
trigger days to the unconditional rest (Welch *t*, Newey-West cross-check), then actually trade
the zone crosses (enter next open, hold 5 days, one-way costs x 2) pinned against a
random-direction control on identical entries — the desk's "beats a coin?" myth-check — and sweep
window x threshold for robustness. A deterministic synthetic tape with a PSY-conditioned reversal
knob proves the harness detects a planted effect and stays quiet on a null. **Dedup:** siblings
[107-stochastic-oscillator](../107-stochastic-oscillator/) (range position, magnitude-weighted),
[127-williams-r](../127-williams-r/) (same range-position family, inverted sign),
[179-aroon](../179-aroon/) (recency-since-extreme, not a frequency count) and
[680-disparity-index](../680-disparity-index/) (close-vs-moving-average deviation) all measure
something PSY deliberately discards — PSY is the pure up/down-day count, and this is that study.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Psychological Line actually counts, why "everyone who wants to buy already has" is a shakier idea than it sounds, the coin test in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the trigger-event design and why it matters, Welch/HAC splits both legs, the window x threshold grid and its parameter-mined corners, the cost sweep, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`psychological_line/`](psychological_line/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
