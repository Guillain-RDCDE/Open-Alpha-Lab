# Study 562 — Block-Trade-Signal 🐋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a giant block print really predict drift? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The claim has real academic support (Easley-O'Hara: informed traders trade *big*, so size carries information) but competes head-on with an equally documented **liquidity-reversal** view (Holthausen-Leftwich-Mayers, Keim-Madhavan: blocks are temporary concessions that snap back) — the *sign* is genuinely disputed. And **no free real tape can even see a classified block** (you'd need the intraday TAQ prints + a Lee-Ready buy/sell classification + survivorship-free returns; daily OHLCV shows none of it), so this is **synthetic-only** and can never earn `REAL`. On the synthetic **null** the coattail is a coin: mean **−0.004%**/event, one-sample *t* **−0.04**, placebo *p* **0.96**, hit **49.5%**. Data-availability limit named on this axis. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | You cannot cheaply *observe* the signal at all — honest block detection needs a paid intraday tape and quote feed. Even the planted-edge illustration shows block-chasing frictions bite: crossing the spread into a dislocated print eats ~40% of a modest edge (gross **+0.42%** → net **+0.25%**/event after 8 bps/leg + a short borrow). Nothing measurable to trade on a retail stack. |

> **In one sentence:** riding the coattails of a giant block is one of the tape's oldest folk beliefs — informed institutions crossing size, drift to follow — but it fights a well-documented *liquidity reversal* (blocks are temporary concessions that revert), and because no free feed can even see a *classified* block this study is synthetic-only: the engine provably catches a planted edge of *either* sign (control mean-*t* −0.13 at the null, up to ±11 as the effect grows), but on the honest null the coattail is a coin (*t* −0.04, placebo *p* 0.96) and there is nothing cheap to observe or trade.

## What we tested

The **block-trade-signal** folklore: a large block-trade *print* on the tape is an *informed-flow*
signal — institutions who know something crossing size — so you can **ride the coattails** (infer
the block's side, take the same side, hold for the drift). The skeptic's view (the microstructure
literature's own finding) is that most large blocks are *liquidity* events negotiated at a
temporary concession that **reverts**. We build a **deterministic synthetic block-print event
panel** (24 stocks × 1 200 events, seed 562) whose single knob `drift_alpha` plants either sign of
the block→forward-return relation (positive = informed drift, negative = liquidity reversal, zero =
null), with a Lee-Ready-style ~15% sign-misclassification folded in. The engine forms the per-event
**coattail return** (copy each block's inferred side, hold the window), and reports a **one-sample
*t*** on that series, a **giant − small block** size sort (two-sample *t*), a **sign-shuffle
placebo** null, a signed-size firm-level slope, a block-size-threshold robustness sweep,
per-event costs + a short borrow, and a **seed-robust (25-seed) synthetic control** proving the
detector catches a planted edge of either sign and stays flat at the null. **This study is
synthetic-only** — the free real data to *see* a classified block does not exist (daily OHLCV has no
prints, no quotes, no side), so per the desk's rubric it is capped at `WEAK`/`NONE` and the
limitation is stated on the Signal axis, like the [lego-returns](../../273-lego-returns/),
[whisky-cask](../../275-whisky-cask/) and [sneaker-resale](../../276-sneaker-resale/) studies.
*Distinct from the daily volume signals [512 High-Volume-Return-Premium](../../512-high-volume-return-premium/)
and [492 Up-Down-Volume](../../492-up-down-volume/): the predictor here is a **single classified
block print** and a directional coattail, not aggregate daily volume; and from
[263 Insider-Buying](../../263-insider-buying/): blocks are anonymous tape prints, not disclosed
insider flow.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a block print is, why chasing it might work (informed flow) or be a trap (liquidity reversal), why we can't even see one cheaply, the honest coin-flip null, and the synthetic control |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the coattail one-sample *t*, the giant−small size sort, the sign-shuffle placebo, the signed-size slope, the size-threshold sweep, costs & borrow, and the seed-robust both-signs synthetic control |

The reproducible synthetic run (null panel fp `cd3cabd17906`, planted-edge panel fp `dc3dd07ca2da`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the deterministic offline generator is
[`block_trade_signal/data.py`](block_trade_signal/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`block_trade_signal/`](block_trade_signal/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
