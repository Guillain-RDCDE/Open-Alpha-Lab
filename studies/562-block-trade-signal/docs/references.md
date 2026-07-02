# References & literature map — Study 562 (Block-Trade-Signal)

## The claim, at full strength (blocks are informed → drift)

- **Easley & O'Hara (1987)**, *"Price, Trade Size, and Information in Securities Markets."*
  *Journal of Financial Economics* 19(1). The theoretical backbone: informed traders prefer to
  trade *large*, so trade *size* itself carries information and market makers move prices more on
  big trades. The formal reason a giant block *could* be an informed-flow fingerprint — the
  `drift_alpha > 0` world.
- **Kraus & Stoll (1972)**, *"Price Impacts of Block Trading on the New York Stock Exchange."*
  *Journal of Finance* 27(3). The seminal block-trade study documenting permanent price effects of
  large blocks — read by folklore as "the block moved it because it knew something."
- **Barclay & Warner (1993)**, *"Stealth Trading and Volatility."* *JFE* 34(3). The
  "stealth-trading" result: informed traders concentrate in *medium* sizes, and a disproportionate
  share of the cumulative price change happens on those trades — a nuance the crude "giant = most
  informed" folklore ignores (and why the size sort here is a genuine test, not a foregone
  conclusion).
- **Chakravarty (2001)**, *"Stealth-Trading: Which Traders' Trades Move Stock Prices?"* *JFE* 61(2).
  Institutional medium-size trades, not the giant prints, do most of the informational work — again
  cutting against the naive coattail story.

## The counter-claim (blocks are liquidity → reversal / the trap)

- **Holthausen, Leftwich & Mayers (1987)**, *"The Effect of Large Block Transactions on Security
  Prices."* *JFE* 19(2). The permanent vs. *temporary* price-impact decomposition: much of a
  block's price move is a **temporary liquidity concession that reverts** — the `drift_alpha < 0`
  world where the coattail loses.
- **Keim & Madhavan (1996)**, *"The Upstairs Market for Large-Block Transactions."* *Review of
  Financial Studies* 9(1). Upstairs-negotiated blocks trade at a price concession to the initiator;
  the temporary component reverses — direct evidence for the reversal side of the sign dispute.
- **Chan & Lakonishok (1995)**, *"The Behavior of Stock Prices Around Institutional Trades."*
  *Journal of Finance* 50(4). Institutional trade *packages* have a large temporary price impact;
  the post-trade drift is far weaker than the folklore assumes once you account for how trades are
  worked over time.

## The classification the coattail needs

- **Lee & Ready (1991)**, *"Inferring Trade Direction from Intraday Data."* *Journal of Finance*
  46(2). The canonical quote rule for signing a trade buyer- vs seller-initiated — the imperfect
  classifier the coattail relies on (and whose ~15% misclassification the synthetic panel folds in).

## Why the sign is the whole question

The two literatures disagree on the *sign* of the block→forward-return relation at the horizon a
retail trader could act on: informed-drift (Easley-O'Hara) vs. liquidity-reversal
(Holthausen-Leftwich-Mayers, Keim-Madhavan). That unresolved sign is precisely why a study with no
real tape must read `WEAK`, not `REAL`: the claim has genuine academic support, but it competes
head-on with an equally well-documented reversal, and only a clean classified-block tape could
adjudicate.

## Why this study is synthetic-only

An honest block-informedness test needs the **intraday TAQ tape** (to see prints far above normal
size), a **Lee-Ready quote-rule classification** (the contemporaneous NBBO to sign each block), and
**survivorship-free forward returns**. Every free, no-key retail feed exposes only daily OHLCV — no
prints, no quotes, no side. A `daily-volume-spike` proxy would inject misclassification and
look-ahead. The desk's rubric caps a synthetic-only study at `WEAK`/`NONE` and names the
data-availability limit on the SIGNAL axis — the same convention as the
[lego-returns](../../273-lego-returns/), [whisky-cask](../../275-whisky-cask/) and
[sneaker-resale](../../276-sneaker-resale/) studies.

## Neighbours on this bench (the dedup map)

- **[Study 512 — High-Volume-Return-Premium](../../512-high-volume-return-premium/)** — the
  daily *volume*-based return premium. Study 562's predictor is a *classified block print* (a single
  large, signed cross), not aggregate daily volume, and the trade is a *directional coattail*, not a
  volume sort.
- **[Study 492 — Up-Down-Volume](../../492-up-down-volume/)** — up- vs down-volume pressure from
  daily bars. Study 562 is about a single *block* event and its inferred *side*, a microstructure
  print, not a daily up/down aggregation.
- **[Study 263 — Insider-Buying](../../263-insider-buying/)** — informed *insider* flow (Form 4).
  Blocks are anonymous tape prints, not disclosed insider transactions; the informedness question is
  the same family but the observable is different.
- **[Study 265 — IPO-Volume](../../265-ipo-volume/)** / **[Study 418 — Money-Flow-Index](../../418-money-flow-index/)**
  — other volume/flow signals; none is a *classified block-print coattail*.

## Shared method

- **One-sample *t*** on the per-event coattail return series — the headline inference bar.
- **Two-sample (Welch) *t*** on the giant − small block spread — the size-informedness test.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: re-sign the
  events' directional component against the return magnitude and read the mean coattail's tail
  probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a **real** tape for `REAL`; synthetic-only caps at `WEAK`/`NONE`), the seed-robust
  synthetic control (≥ 20 seeds), one execution lag, and costs one-way × NAV with shorts paying
  borrow.
