# Study 15 — Sigma-Sleight 📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do length-aware σ-zones read RSI better than fixed 70/30? | 🟡 `WEAK` | The motivating fact is **real**: fixed 70/30 *is* length-naive (a short RSI swings 0–100, a long one hugs 50). But the σ-transform sold as the fix — `σ = logit(RSI)·√(n−1)/2` — is **strictly monotone**, so within a length it merely *renames* a constant RSI level: on real SPY/QQQ the "adaptive zone" and a plain fixed band trade **identically** (max position diff **0** across all 6 length×ticker cells). And the σ-calibrated band beats naive 70/30 in only **1/6** cells — its own levels are *more* extreme (RSI(2) `−√3σ` = RSI 3.0), so they trade worse, not better. The only genuine content is a re-statement of RSI arithmetic. |
| **Tradability** — could you get paid? | 🔴 `MIRAGE` | Even a surviving threshold edge is a **daily RSI mean-reversion rule** paying an equity spread every round-trip to express a few points of tilt — and the framework's own hedge is that its zones are *"not buy/sell signals."* On the real tape the σ-band beats a re-optimised constant in **0/6** cells (ΔSharpe −0.21 to −0.68), because it *is* one constant among many. |
| **Does the σ-transform add signal?** | ⚪ `RELABEL` | Monotone ⇒ order-preserving ⇒ every threshold crossing and rank statistic is invariant. On the real tape Rescaled RSI(70) carries the **exact** rank IC of raw RSI(70) (gap `0.0e0`); the cheat-sheet (RSI(14)=70 ⟺ **+1.53σ**) is a function of `n` alone, no market data. |

> **In one sentence:** AdaptiveRSI's length-awareness is a **genuinely sound idea wrapped in a σ-transform that, being strictly monotone, only relabels** — within a length it renames a constant RSI threshold and moves zero trades (max crossing diff 0 on real SPY/QQQ), the famous cheat-sheet is pure arithmetic, "Rescaled RSI" is rank-identical to the raw long RSI, and the σ-calibrated band beats naive 70/30 in just 1 of 6 real cells while never beating a re-optimised constant; the σ apparatus is a better *vocabulary*, not an edge it creates.

## What we tested

The [AdaptiveRSI framework](https://adaptiversi.gumroad.com) ([TradingView](https://www.tradingview.com/), *"RSI Beyond 70/30"* + a free 38-page *"RSI Manifesto"*) argues that the classic **70/30** overbought/oversold lines are length-naive — RSI(2) at 70 and RSI(200) at 70 are not the same event — so thresholds should be defined as standardized **σ landmarks** in a logit-RSI space and translated back per length via a `2/√(n−1)` factor, under which **RSI(14)=70 lands at +1.53σ**. We steelman it (fixed 70/30 *is* length-naive, and short-period RSI mean reversion is a real, modest effect), then test the one falsifiable thing it implies — that σ-zones read extremes *better* than 70/30 — on daily SPY/QQQ closes, splitting the claim into two **exact identities** (the σ-transform and Rescaled RSI are monotone relabels) and one empirical horse race (σ-band vs fixed 70/30 vs a re-optimised constant, net of costs).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language, why a strictly-rising curve gives the game away, and the constant hiding inside every "adaptive" zone |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: monotone-bijection proof, the crossing-identity and rescale-invariance (both exactly 0), the arithmetic cheat-sheet, and the cost-charged horse race vs a re-optimised constant |

The real run — every fingerprinted, as-of'd horse-race table — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (a *genuinely* vol-adaptive σ-band — the framework's unbuilt regime promise — which moves the threshold several RSI points yet beats a plain fixed level in 0 of 6 real cells) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the daily-close cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
