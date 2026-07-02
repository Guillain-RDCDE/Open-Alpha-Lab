# Results — Study 562 (Block-Trade-Signal): can you ride the coattails of a giant block, on a synthetic tape

*Generated from [`block_trade_signal/`](../block_trade_signal/). **Synthetic-only study** — there is
no free, no-key feed of *classified* block trades (the intraday TAQ tape + a Lee-Ready buy/sell
classification + survivorship-free forward returns; see the [data note](#why-synthetic-only)), so
this study builds a deterministic synthetic block-print event panel and never touches a real tape.
The panel is a 24-stock × 1 200-event cross-section (seed 562). Two worlds share one schema: the
**null** (no block→drift relation, `drift_alpha = 0`, panel fp `cd3cabd17906`) and a
**literature-modest planted edge** (`drift_alpha = 0.006`, panel fp `dc3dd07ca2da`). As-of
**2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The folklore: when a **giant block crosses the tape**, it is an informed institution crossing size,
and the stock will *drift* in the direction of that informed flow — so you can *ride the coattails*
(infer the block's side, take the same side, hold for the drift). The microstructure literature's
own finding cuts the other way: most large blocks are *liquidity* events negotiated upstairs at a
temporary price concession that **reverts** (Holthausen-Leftwich-Mayers 1987; Keim-Madhavan 1996;
Chan-Lakonishok 1995), so the coattail either earns nothing or *loses* after the spread you pay to
chase a dislocated print. The *sign* is genuinely contested.

We cannot settle it on a real tape, because **the real data a retail stack can reach cannot see a
block at all** — every free feed exposes only daily OHLCV, with no intraday prints, no quotes, and
no trade-direction sign. So this is a **synthetic-only** study: the machinery is proven on a
deterministic panel, and per the desk's rubric a synthetic-only study can never earn `REAL` (that
needs a robust *t* ≥ 2 on a real tape). The literature genuinely supports *a* block-informedness
relation, but with no real-tape certification and a live sign dispute (informed drift vs. liquidity
reversal), that reads **`WEAK`** on the signal axis; and because you cannot cheaply *observe* the
signal at all on a free stack, it is `MIRAGE` on tradability.

## The honest baseline — the null world (no planted edge)

When there is **no** block→drift relation (`drift_alpha = 0`), the engine correctly finds nothing:

| | value |
|---|---|
| Mean coattail return per event (copy the block's side, hold the window) | **−0.004%** |
| One-sample *t* on the coattail series (1 200 events) | **−0.04** |
| Event hit rate | **49.5%** (≈ coin) |
| Label-shuffle (sign) placebo *p* | **0.964** |
| Giant − small block coattail spread (terciles) | **+0.14%** (two-sample *t* +0.59) |
| Signed-size slope *t* (bigger block → more drift) | **−0.13** (corr −0.004) |

This is what "a block print tells you nothing" looks like: a coattail P&L indistinguishable from
zero, a placebo *p* near 1, no size effect, no firm-level slope. It is the null the real world
*might* be — and we have no real tape to rule it out.

## The literature-modest planted edge (illustration only)

If we plant a small, realistic informed-block edge (`drift_alpha = 0.006` — correctly-classified
informed blocks drift the coattail's way in proportion to their size), the engine recovers it:

| | value |
|---|---|
| Mean coattail return per event (gross) | **+0.419%** |
| One-sample *t* (1 200 events) | **+4.32** |
| Event hit rate | **55.0%** |
| Label-shuffle placebo *p* | **0.0005** |
| Giant − small block coattail spread | **+0.62%** (two-sample *t* +2.50) |
| Signed-size slope *t* | **+2.09** (corr +0.060) |
| Net (8 bps/leg round-trip + 100 bps/yr borrow on the short side, 5-day hold) | **+0.249%** |

This is an **illustration of a planted world, never evidence for the tape** — a synthetic panel
with an effect switched on will of course be significant. It shows only that the detector works, and
that block-driven frictions (crossing the spread into a dislocated print) chew roughly **40%** of a
modest edge (gross +0.42% → net +0.25% per event) even before you count slippage and capacity.

## Robustness — size threshold (planted world)

The folklore is strongest for the *biggest* blocks. Restrict the coattail to prints above a rising
size cut:

| Min block size (× normal) | Mean coattail | *t* | events | hit |
|---|---|---|---|---|
| ≥ 0.0 (all) | +0.42% | +4.32 | 1 200 | 55.0% |
| ≥ 1.5 | +0.57% | +4.49 | 713 | 56.9% |
| ≥ 2.0 | +0.71% | +4.28 | 439 | 59.2% |
| ≥ 3.0 | +0.87% | +3.09 | 159 | 62.3% |

In the planted world the effect *strengthens per event* as blocks get more giant (the drift scales
with size) while the *t* softens as the sample shrinks — the sign is stable throughout. In the
**null** world the same sweep wanders around zero (*t* −0.04, +0.21, +0.25, −0.20), the signature of
noise.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

The house rule: average the coattail *t* over ≥ 20 seeds so no single lucky seed manufactures a
result. The knob plants either sign of the block effect (informed drift vs. liquidity reversal):

| Planted `drift_alpha` | Mean coattail-*t* (25 seeds) | Reads as |
|---|---|---|
| −0.020 | **−11.44** | strong reversal (liquidity blocks) |
| −0.012 | −8.13 | reversal |
| −0.006 | −4.51 | reversal (the trap) |
| **0.000 (null)** | **−0.13** | flat — no false signal |
| +0.006 | +4.26 | informed drift |
| +0.012 | +7.92 | informed drift |
| +0.020 | +11.28 | strong informed drift |

At the null the mean *t* is ≈ 0; planting informed drift drives it positive and planting the
liquidity reversal drives it negative — **the engine catches both signs and stays flat at zero**. So
the verdict is a statement about the *evidence available* (synthetic-only, no real tape), not a
broken detector.

## Why synthetic-only

An honest block-informedness test needs three things no free retail feed provides:

1. **The intraday tape.** To flag a "block" you need the trade-by-trade prints (TAQ), so you can see
   a single cross far above the prevailing trade size. Daily OHLCV shows a *volume* number, not
   individual prints — a high-volume day is not a block, and cannot be turned into one.
2. **A buy/sell classification.** The coattail needs the block's *side*. That is a Lee-Ready-style
   quote rule comparing each print to the *contemporaneous* NBBO — which needs the microsecond
   best bid/offer, another paid TAQ field. Daily bars carry no side.
3. **Survivorship-free forward returns.** The drift is measured over the days after the block; a
   real study needs delisting-adjusted returns.

A `daily-volume-spike` proxy hacked from OHLCV would inject exactly the misclassification and
look-ahead the study must avoid. Rather than publish a dishonest "real" number, this study is
synthetic-only and says so on the SIGNAL axis (like the desk's
[lego-returns](../../273-lego-returns/), [whisky-cask](../../275-whisky-cask/) and
[sneaker-resale](../../276-sneaker-resale/) studies).

## The honest takeaway

Riding a giant block's coattails is one of the tape's oldest folk beliefs, but it fights a
well-documented *reversal* view (blocks are liquidity concessions that snap back), and **no free
real tape can even see a classified block, let alone settle the sign.** On the synthetic null the
coattail is a coin (mean −0.004% per event, *t* −0.04, placebo *p* 0.96, hit 49.5%); the engine
provably catches a planted edge of either sign (control: −0.13 at the null, up to ±11 as the effect
grows), and block-driven frictions eat ~40% of even a modest planted edge. `WEAK` (literature says
maybe-real; no real-tape certification; sign disputed) × `MIRAGE` (you cannot cheaply observe the
signal to trade it).
