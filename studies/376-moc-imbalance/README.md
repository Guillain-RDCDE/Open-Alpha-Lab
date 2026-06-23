# Study 376 — MOC-Imbalance 🔔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the closing push reverse overnight? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The reversal slope is *direction-correct* (negative: a push into the close does lean back next morning) but **fails t ≥ 2** on the real tape (HAC **t = −1.48**, placebo *p* = **0.076**), explains **~0.01%** of the overnight gap's variance, and — decisively — **weakens on the index-rebalance days** where the lore says it should be strongest (**t = −0.48**). A correct-sign whisper inside its own noise, on a proxy. |
| **Tradability** — can you fade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Gross fade is **+0.18 bps/trade** (daily Sharpe ≈ **0.04**); a **1-bp** round-trip turns it **negative**, 2 bps loses **−4.5%/yr**. The "snap-back" lives **inside the spread** — captured by the auction's liquidity providers, paid by anyone who crosses it. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | Real-as-microstructure, empty-as-alpha: a tiny correct-sign effect the bid-ask spread eats whole, that **evaporates precisely on the rebalance days** believers point to. |

> **In one sentence:** the closing-auction-imbalance reversal trade — fade the push into the close, pocket the overnight snap-back — shows up on a transparent end-of-day-pressure proxy with the *right sign* but a HAC *t* of only −1.48 (and a near-zero R²), it gets *weaker* on the index-rebalance days where a real MOC imbalance should bite hardest, and even gross the fade earns 0.18 bps a trade — a rounding error the bid-ask spread turns negative — so it is real-as-microstructure, weak-as-edge, and undeployable.

## What we tested

There is **no free closing-auction-imbalance feed** (the NYSE/Nasdaq MOC imbalance message is a paid product), so we build a **transparent end-of-day-pressure proxy** from yfinance daily bars: each day's signed intraday **displacement** `(close − open)/(high − low)` — +1 means price opened at the low and closed at the high, the maximum upward push into the close. We then test the believers' question directly: does a big push **reverse overnight** (a negative close→open gap the next morning)? We regress the overnight gap on the displacement with an autocorrelation-robust **HAC (Newey-West) t-stat**, a placebo randomization null, and a fade strategy with a structural 1-day lag and one-way costs — on the full tape **and** on the **index-rebalance subset** (third Friday of quarter-end months), where real MOC imbalances are largest. A deterministic synthetic control plants a *known* overnight-reversal edge to prove the engine is faithful (edge = 0 stays insignificant; a large planted edge lights up at t = −8). Same microstructure-eaten-by-frictions pattern as [Study 140](../140-amihud-illiquidity/), and a slice of the overnight-vs-intraday split in [Study 01](../01-overnight-anomaly/).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a closing auction is, why "the imbalance pushes the close and reverses overnight" sounds like free money, and why the snap-back is really inside the spread — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the displacement proxy, an `overnight ~ disp` reversal regression with a HAC *t* + placebo null, the rebalance-day subset, the fade strategy net of costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`moc_imbalance/`](moc_imbalance/). Closing-auction pressure here is an explicit **proxy** (intraday displacement), not a true MOC order-imbalance feed. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
