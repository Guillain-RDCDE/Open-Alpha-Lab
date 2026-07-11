# Study 668 — Williams-VIX-Fix

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the price-only "VIX Fix" spike ahead of real bounces? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pooled 8-ticker gap vs unconditional: **+9.6 bps** (5d, *t* = +1.33), **−8.8 bps** (10d, *t* = **−0.88**), **−23.9 bps** (20d, *t* = **−1.74**) — no horizon clears *t* ≥ 2, the sign flips negative by 10 days, and a 10-day random-calendar placebo shows the observed mean losing to **87.4%** of random calendars. |
| **Tradability** — does it survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The 10-day spike-onset timer doesn't clear *t* ≥ 2 even **gross** (+1.76); 5 bps one-way costs drop it to *t* = +1.17, 10 bps to *t* = +0.58. There's no edge left for costs to eat. |
| **More than a drawdown proxy?** | ![Busted](https://img.shields.io/badge/More_than_a_drawdown_proxy%3F-Busted-8b949e?style=flat-square) | 75.5% of every WVF spike is a day a plain **close-only** "far below its 22-day high" flag would have caught anyway; the intrabar low's marginal contribution (HAC, controlling for that plain proxy) averages *t* = **−0.30** across the basket — the wick buys nothing. |

> **In one sentence:** Larry Williams' price-only "VIX Fix" — `(highest_close_22 − low) /
> highest_close_22 × 100` — does spike hard exactly when price wicks far below its recent
> highs, but the spike carries no forward-return edge at 5, 10 or 20 days (basket Welch
> *t* = +1.33 / −0.88 / −1.74, a 10-day placebo the observed mean *loses*), the "trade" is
> a Mirage before costs are even charged, and a head-to-head HAC test shows the formula's
> one extra ingredient over a plain drawdown filter — the intraday low — adds nothing.

## What we tested

Larry Williams' VIX Fix turns any OHLC bar series into a "synthetic VIX" with no options
data required: `WVF = (highest_close(22) - low) / highest_close(22) * 100`, popularized on
retail platforms as the "CM_Williams_Vix_Fix" indicator — spike above a Bollinger band on
itself (20-session mean + 2σ), call it capitulation, buy. We test the spike's onset against
forward returns at 5/10/20 days across an eight-ticker basket (SPY/QQQ/IWM — no
survivorship — plus AAPL/MSFT/JPM/XOM/JNJ — survivorship named), pool the basket for the
primary Welch *t* and cross-check per ticker with a Newey-West regression (lags = horizon,
because overlapping forward windows are mechanically autocorrelated), run a random-calendar
placebo, price a "buy the fear spike" timer at 5/10 bps one-way costs, and ask the question
a fair review asks first: is any of this more than "price fell a lot," by pitting WVF
head-to-head against a plain close-only drawdown proxy in a two-dummy HAC regression.
**Dedup:** siblings [111-vix-term-structure](../111-vix-term-structure/) (the real VIX
curve slope), [92-easy-money](../92-easy-money/) (VIX-futures carry),
[127-williams-r](../127-williams-r/) (Larry Williams' other, unrelated oscillator) and Bill
Williams' [184-williams-fractals](../184-williams-fractals/) /
[421-williams-alligator](../421-williams-alligator/) never test a price-only VIX proxy or
whether it beats a plain drawdown signal — this study's own two axes. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "fear gauge built from price alone" even means, why the spike looks so convincing on a chart, and why it still doesn't pay |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the horizon sweep, the per-ticker Newey-West cross-check, the random-calendar placebo, the drawdown-proxy head-to-head, the cost-adjusted timer and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`williams_vix_fix/`](williams_vix_fix/). Three of eight basket names are broad
index ETFs (no survivorship); the other five are named survivors of a long clean tape (see
[docs/references.md](docs/references.md)). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
