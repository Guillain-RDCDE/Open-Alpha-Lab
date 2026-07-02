# References & literature map — Study 586 (Liquidation-Cascade)

## The claim under test

The **capitulation-bottom / buy-the-blood** folklore: a large wave of **forced liquidations** in
crypto perpetual-futures markets — over-leveraged longs margin-called and closed by the exchange en
masse — is *mechanical, non-informational* selling that overshoots the price to the downside, so a
big liquidation spike marks a local bottom and the price then **bounces**. The tradable read is
"buy after the liquidation cascade." It is desk/Twitter folklore rather than a single canonical
paper, but it rests on a real microstructure literature on fire-sale price pressure and its
reversal.

## The microstructure it leans on

- **Shleifer, A. & Vishny, R. (1992/2011)**, *"Fire Sales in Finance and Macroeconomics."* The
  foundational fire-sale idea: forced, liquidity-motivated selling pushes prices below fundamentals
  when natural buyers are themselves constrained — the temporary-overshoot mechanism the
  capitulation-bounce story assumes.
- **Coval, J. & Stafford, E. (2007)**, *"Asset Fire Sales (and Purchases) in Equity Markets."*
  *JFE*. Forced mutual-fund selling causes measurable, *temporary* price pressure that subsequently
  **reverses** — the cleanest equity-market analogue of "mechanical selling overshoots, then
  bounces."
- **Brunnermeier, M. & Pedersen, L. (2009)**, *"Market Liquidity and Funding Liquidity."* *RFS*.
  Margin/funding spirals: a price drop tightens margins, forcing more selling, feeding the drop —
  the *cascade* half of "liquidation cascade." Whether the overshoot then reverses is the empirical
  question this study frames.
- **Nagel, S. (2012)**, *"Evaporating Liquidity."* *RFS*. Short-horizon reversal strategy returns
  spike exactly when liquidity evaporates (VIX high) — i.e. the reversal-after-forced-selling
  premium is real but concentrated in stress, which is where a liquidation-event filter would fire.

## Crypto-specific context

- **Perpetual-futures liquidation mechanics.** On Binance/Bybit/OKX-style venues, a position that
  breaches its maintenance margin is *liquidated* by the exchange's engine (often via an insurance
  fund / auto-deleveraging), producing a burst of forced market orders. Aggregate per-day
  liquidation-USD is the natural event series — and is a **paid** data product (Coinglass,
  Amberdata, exchange futures APIs), with no usable *free* history. This is the data-availability
  wall that makes the study synthetic-only and caps it at `WEAK`/`NONE`.
- **Reversal vs. momentum in crypto.** The desk's own [251 crypto-reversal] and [210 crypto-trend]
  show that short-horizon crypto reversal is weak-to-absent net of costs, while trend persists at
  some horizons — the two directions a post-liquidation move could take (bounce vs. falling knife),
  which is exactly the sign the ``bounce_alpha`` knob controls.

## Neighbours on this bench (the dedup map)

- **[133 crypto-seasonality]** / **[175 crypto-weekend]** — *calendar* effects on the crypto tape;
  Study 586 keys on *forced-liquidation flow*, a microstructure channel, not the calendar.
- **[210 crypto-trend]** / **[251 crypto-reversal]** — *price-only* momentum/reversal on the tape;
  586 conditions the forward return on a *liquidation-event* flag, not on past returns.
- **[325 crypto-fear-greed]** — a *sentiment* index as the conditioning variable; 586's
  conditioner is *realised forced-liquidation USD*, a hard-flow microstructure quantity.

## Shared method

- **Event study.** Mean forward return on flagged (large-liquidation) days minus the unconditional
  baseline, with the same-day (mechanical) crash excluded by construction — the standard
  event-window design (MacKinlay 1997, *"Event Studies in Economics and Finance"*, JEL).
- **Welch (1947)** — the unequal-variance two-sample *t* for event vs non-event forward returns.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  event labels against the fixed forward returns and read the gap's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a **real** tape for `REAL`; synthetic-only studies are capped at `WEAK`/`NONE`), one
  documented execution lag (enter at the event close, hold the *next* H days), costs one-way × NAV,
  and the seed-robust (≥ 20 seeds) synthetic positive control for any synthetic-dependent claim.
