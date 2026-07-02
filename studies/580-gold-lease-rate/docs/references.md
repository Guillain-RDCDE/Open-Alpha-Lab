# References & literature map — Study 580 (Gold-Lease-Rate)

## The claim & the microstructure

- **LBMA — GOFO (Gold Forward Offered rate).** The London Bullion Market Association published a
  daily GOFO benchmark (the rate to swap gold for USD dollars via a forward) from which the
  **gold lease rate** was implied as `lease ≈ LIBOR − GOFO`. The LBMA **ceased publishing GOFO on
  30 January 2015**, removing the one clean, continuous public source of an implied lease rate —
  the reason this study is synthetic-only.
- **Gold lease rate / gold-forward market microstructure.** A positive lease rate means holders of
  bullion are paid to lend it; spikes (physical-scarcity scrambles, backwardation) are read by
  practitioners as a leading indicator of price pressure. See practitioner treatments of gold
  leasing, GOFO and the forward curve (e.g. LBMA guides; World Gold Council commentary on the
  gold forward market). The *lead-lag* claim — that the borrow cost foreshadows the spot price —
  is **folklore**: widely repeated, rarely tested cleanly out of sample.
- **Commodity carry / convenience yield.** The lease rate is the gold analogue of a *convenience
  yield* / carry. Gorton & Rouwenhorst (2006), *"Facts and Fantasies about Commodity Futures"*
  (Financial Analysts Journal) and the broad commodity-carry literature (e.g. Koijen, Moskowitz,
  Pedersen & Vrugt 2018, *"Carry"*, JFE) establish carry as a cross-sectional predictor of
  commodity returns — the intellectual backdrop for a *time-series* lease-rate → gold claim.

## Neighbours on this bench (the dedup map)

- **[Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/)**,
  **[Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/)**,
  **[Study 388 — Lumber-Gold-Ratio](../../388-lumber-gold-ratio/)** — cross-asset *price ratios*
  as timing signals. Study 580 is a **microstructure carry/borrow** signal (the cost to borrow
  bullion), not a price ratio.
- **[Study 208 — Gold-Miners](../../208-gold-miners/)** — the leveraged equity beta on gold; a
  different instrument entirely.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)** / **[Study 70 — Digital-Gold](../../70-digital-gold/)**
  — other gold-adjacent folklore; distinct claims.

## Shared method

- **Predictive (lead-lag) regression** — OLS of the forward return on a *lagged*, already-public
  predictor; the sign and *t* of the slope are the claim. The one-period lag is the single
  documented execution convention.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  lease-rate predictor against forward gold returns and read the |t| tail probability.
- **Seed-robust synthetic positive control** — averaging the test *t* over ≥ 20 seeds so no single
  lucky RNG seed can manufacture significance (house rule for synthetic-dependent claims).
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a **real** tape for `REAL`; synthetic-only caps at `WEAK`), one documented execution
  lag, gross/net labelling, and the data-availability caveat on the SIGNAL axis (as in the desk's
  other synthetic-only studies — [273 lego-returns](../../273-lego-returns/),
  [275 whisky-cask](../../275-whisky-cask/), [276 sneaker-resale](../../276-sneaker-resale/)).
