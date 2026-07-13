# References & literature map — Study 768 (Charm-Decay)

## The claim under test

*Charm* (also "delta decay" or "delta bleed") is the second-order option Greek
∂Δ/∂t — the rate at which an option's delta changes purely because time passes, holding the
underlying fixed. For near-the-money options it is largest in the final days before expiry.
The popular options-flow narrative — propagated by dealer-positioning research desks and
vol-Twitter — holds that in the last week before **monthly** options expiration (the 3rd
Friday), the aggregate delta of the dealer book bleeds predictably, forcing systematic
re-hedging that (a) pushes the underlying **up into OpEx** ("the OpEx-week rally") and (b)
reverses into **weakness the week after** as the flow unwinds and the following cycle's
positioning resets. This study asks whether that directional drift exists in the SPY tape.

## Where the story comes from (the steelman)

- **SpotGamma / dealer-positioning research** — popularised the "charm and vanna flows"
  framing of index drift into monthly OpEx: as expiration approaches, decaying deltas on the
  large put-heavy dealer book mechanically generate supportive buy-to-hedge flow, released
  the week after. (Educational notes and market commentary, 2020–.)
- **Menthor Q, Kai Volatility, and the vol-Twitter ecosystem** — routinely attribute
  pre-OpEx equity strength and post-OpEx softness to charm/vanna hedging, especially in
  low-realised-vol regimes where dealer gamma is long and stabilising.
- **Brogaard, Han & Won (2023 wp), *How Does Options Hedging Impact the Underlying?*** —
  academic treatment of the mechanism by which dealer delta-hedging of option inventory
  transmits to underlying price dynamics; the microstructure channel the folk claim invokes.

## Theoretical foundations of the Greek

- **Black & Scholes (1973)**, *The Pricing of Options and Corporate Liabilities* (Journal of
  Political Economy) — the delta-hedging framework from which charm (∂Δ/∂t) is derived as a
  higher-order sensitivity; hedgers who are short options must rebalance as the deltas move,
  including the pure time-decay component.
- **Taleb (1997)**, *Dynamic Hedging: Managing Vanilla and Exotic Options* (Wiley) — the
  practitioner reference that named and popularised the "minor" Greeks charm, vanna, and
  colour, and the mechanics of hedging them near expiry.
- **Ni, Pearson & Poteshman (2005)**, *Stock Price Clustering on Option Expiration Dates*
  (Journal of Financial Economics) — the canonical evidence that option hedging measurably
  moves the underlying around expiry (prices are pulled toward strikes). It documents a
  *pinning* (variance-suppressing) effect, not a *directional drift* — a useful contrast: the
  hedging channel is real, the directional-drift claim is the extrapolation this study tests.

## Related expiration-effect evidence

- **Stoll & Whaley (1987)**, *Program Trading and Expiration-Day Effects* (Financial Analysts
  Journal) — elevated volume and volatility on quarterly triple-witching; the classic
  expiration-day literature the charm claim tries to generalise into a directional monthly signal.
- **Gârleanu, Pedersen & Poteshman (2009)**, *Demand-Based Option Pricing* (Review of
  Financial Studies) — end-user option demand imbalances feed back to underlying prices via
  dealer hedging; the pricing channel behind vanna/charm flow stories.
- **Bollen & Whaley (2004)**, *Does Net Buying Pressure Affect the Shape of Implied Volatility
  Functions?* (Journal of Finance) — evidence that dealer net positioning shapes the option
  surface, a precondition for a systematic hedging-flow drift.

## Related desk studies (shared tape and method)

- **[Study 195 — Monthly-OpEx](../../195-monthly-opex/)** — tests the OpEx *week* on the same
  SPY tape for volume, range, and return; finds the volume uplift is quarterly triple-witching
  only and the return is null (Signal NONE, Tradability MIRAGE). This study is its directional
  companion: not "is there activity?" but "is there a tradable charm-driven *drift*?"
- **[Study 82 — Witching-Hour](../../82-witching-hour/)** — the quarterly triple-witching
  parent effect (volume real, return weak, untradable).
- **[Study 370 — Zero-DTE-Options](../../370-zero-dte-options/)** and
  **[Study 111 — VIX-Term-Structure](../../111-vix-term-structure/)** — same options/vol
  microstructure family; overlay-vs-SPY, HAC inference, costs-net, capacity-noted framework.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  every window *t* here uses it ([`strategy.compare_window_vs_baseline`](../charm_decay/strategy.py)).
- **Calendar-randomisation placebo.** A permutation/placebo null built by displacing the
  event anchor — the standard falsification for any calendar-anchored effect
  ([`strategy.placebo_randomization`](../charm_decay/strategy.py)). Compare Lo & MacKinlay
  (1990) on data-snooping in seasonally-defined trading rules.
- **Multiple comparisons / inference bar.** Harvey, Liu & Zhu (2016), *… and the Cross-Section
  of Expected Returns* (Review of Financial Studies) — argues for elevated *t*-thresholds
  given the search intensity; here the effect does not even clear the unadjusted |t| = 2.
