# Study 670 — Bollinger-Squeeze

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout call the direction? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Enter next open in the sign of the release-bar slope, hold 10 days: **−36.5 bps/trade** net at 5 bps costs, vs a matched random-timing/same-direction-mix control at **−44.4 bps** — Welch *t* = **+0.15**. **0 of 27** BB-std × KC-mult × hold-day combinations clear \|*t*\| ≥ 2. No ticker beats its own random control either way. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The ledger is negative net of costs and statistically identical to picking the same long/short calls on random dates. There is no edge to scale — only costs to pay for noise. |
| **Does the squeeze even predict MORE volatility than a random day?** | ![Busted](https://img.shields.io/badge/Vol_expansion%3F-Busted-8b949e?style=flat-square) | Pooled forward-10-day realized vol after a release vs a **matched random day**: Welch *t* = **+0.12** — no incremental timing information. The "vol WILL expand" story only holds in the trivial sense of reverting off the squeeze's own contracted baseline (significant only for SPY; it *inverts* for GLD, *t* = −2.30) — not in the sense a trader needs. |

> **In one sentence:** the TTM Squeeze correctly notices that Bollinger-inside-Keltner marks an
> unusually quiet moment — but across 111 squeeze-release events on SPY/QQQ/IWM/DIA/GLD
> (2005–2026) the subsequent volatility expansion carries **no more timing information than a
> random day** (Welch *t* = +0.12), the breakout direction is a **coin flip** against a matched
> random control (*t* = +0.15, 0/27 parameter combos clear the bar), and net of costs the trade
> **loses money indistinguishably from noise** — a clean `None x Mirage`.

## What we tested

The claim, steelmanned the way TTM Squeeze traders state it: when the Bollinger Band (SMA20 ±
2σ) contracts to sit entirely inside the Keltner Channel (EMA20 ± 1.5·ATR20) for several bars,
realized volatility has compressed to an unusual low, and the eventual re-expansion — the
"squeeze release" — signals a big directional move about to happen; trade the breakout. We
detect every release (≥ 5 consecutive squeeze bars, then the first bar the condition fails) on
five liquid daily ETF tapes (SPY, QQQ, IWM, DIA, GLD; yfinance, 2005 → 2026), split the claim
into its **two distinct halves** — does volatility genuinely expand more than on a random day,
and does the breakout direction (a causal trailing-slope sign) call the next move profitably —
and pin both against **matched random controls** (random-day for the vol test, random-timing
with the identical direction mix for the directional test) so generic drift and everyday
volatility clustering never get mistaken for squeeze-specific information. A parameter sweep (27
band/hold combinations) and a deterministic synthetic positive control (a genuine mean-reverting
range-bound regime with a tunable planted continuation effect) confirm the null result is a real
"nothing there," not a blind spot in the machinery. **Dedup:**
[104-bollinger-reversion](../104-bollinger-reversion/) (the opposite, single-band reversion
trade), [128-keltner-channel](../128-keltner-channel/) (Keltner alone, no Bollinger comparison),
[485-starc-bands](../485-starc-bands/) (a third, unrelated envelope) and
[190-nr7](../190-nr7/) (a single-bar range-contraction signal) never test the two-indicator
BB-inside-KC squeeze geometry or separate the vol-expansion claim from the directional claim the
way this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a squeeze is, why "vol expands after a squeeze" sounds like proof but mostly isn't, the coin-flip breakout, why the trade loses to random |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the random-day vol test, the random-timing/direction-mix control, the per-ticker and parameter-sweep breakdowns, costs, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bollinger_squeeze/`](bollinger_squeeze/). Bands are causal (trailing data only);
signal on the release bar's close, entered at the next bar's open (one documented lag). Basket
is long-lived, still-listed ETFs — no survivorship panel. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
