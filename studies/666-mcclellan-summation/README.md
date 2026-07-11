# Study 666 — McClellan Summation Index 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the running integral of breadth time SPY? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The textbook "crosses zero → confirms the regime" rule **never fires** — 0 post-warm-up zero-crosses, in either direction, over 21.4 years and every EMA-span/basket variant tried. A scale-appropriate ±1σ analog of the literature's ±500 extreme shows no correctly-signed edge over a drift-matched random-day baseline at any of 4 horizons (7/8 Welch *t*'s indistinguishable from noise; the one exception, 60d, is **negative** — the wrong sign), and a shuffled-breadth placebo confirms the breadth path carries no information (*p* = 0.309). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The only regime rule that actually switches (the z-score adaptation, since the textbook zero level never crosses) **underperforms buy-and-hold by −3.1 bps/day** (HAC *t* = **−2.43**, robust across EMA spans and baskets) — but the regime-timer placebo (*p* = 0.491) shows this is generic drift give-back from sitting out of the market sometimes, not a real anti-signal either. |
| **"Does the Summation Index confirm bull/bear turns?"** | ![Busted](https://img.shields.io/badge/Confirms_turns%3F-Busted-8b949e?style=flat-square) | The zero level never confirms anything (it never crosses); the extreme-threshold analog doesn't beat random days; the regime timer built on it loses to simply holding the index. |

> **In one sentence:** the McClellan Summation Index — the running cumulative sum of the
> oscillator studied in [491](../../491-mcclellan-oscillator/) — turns out to have a fatal
> structural flaw for a small ETF-basket proxy: the un-rebased integral drifts away from zero
> in its first few months and **never comes back** across 21.4 years, so the textbook
> zero-cross regime signal cannot fire even once, its scale-adapted extreme-threshold cousin
> doesn't beat random days, and the resulting long/flat timer loses money to buy-and-hold
> (HAC *t* = −2.43) for reasons a shuffled-breadth placebo shows are generic, not informative.

## What we tested

We build the Summation Index causally — `cumsum(EMA₁₉(net_adv) − EMA₃₉(net_adv))` on a daily
net-advances proxy across SPY + the 9 classic SPDR sector ETFs (2005→2026, no survivorship:
the actual traded universe throughout) — and test the claim three ways: (1) the literal
**zero-cross** regime-turn signal, event-studied against a drift-matched random-entry
baseline; (2) a causal **rolling z-score** ±1σ cross as the honest, scale-appropriate analog
of the literature's ±500 level (a 10-name basket cannot reproduce a full-NYSE numeric
threshold); and (3) a **long/flat regime timer** with one documented execution lag and 5 bps
costs per switch, its excess over buy-and-hold tested with a Newey-West (HAC) *t*. A
shuffled-breadth placebo (time-permute the net-advances series, rebuild the index, re-fire
the rule) checks whether the specific breadth-momentum geometry is load-bearing in either
direction, and a 20-seed deterministic synthetic control proves the harness can bank a
planted regime effect it is given. **Dedup:** sibling
[491-mcclellan-oscillator](../../491-mcclellan-oscillator/) tests the oscillator's own
single-day trigger; [494-bullish-percent-index](../../494-bullish-percent-index/),
[168-advance-decline](../../168-advance-decline/) and
[493-new-highs-new-lows](../../493-new-highs-new-lows/) test other breadth constructions
entirely. None of them test the Summation Index's own zero-cross / extreme-level / long-flat
claim — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Summation Index is, why "just add up the oscillator" sounds smart, why the running sum gets structurally stuck away from zero, and what that does to the regime-timing trade — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the causal cumsum construction, the zero-cross non-event, the rolling-z-score threshold design, the random-entry and shuffled-breadth placebos, the HAC-tested regime timer, EMA-span/basket robustness, and a 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`mcclellan_summation/`](mcclellan_summation/). The Summation Index is a causal,
never-re-based running sum of EMA₁₉−EMA₃₉ of net advances; the regime is read on close *t-1*
and applied to day *t*'s return (one lag). Breadth is a **proxy** built from a 10-name
liquid-ETF basket (no survivorship — the actual traded universe throughout). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
