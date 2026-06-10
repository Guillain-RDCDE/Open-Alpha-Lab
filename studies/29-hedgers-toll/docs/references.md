# References & literature map — Study 29 (Hedgers-Toll)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §9.2 (trading based on
  hedging pressure)** — use the net positioning of commercial hedgers to predict commodity futures
  returns.

## The claim under test — the steelman

- **Normal backwardation / hedging pressure.** John Maynard Keynes, *A Treatise on Money* (1930) and
  John Hicks, *Value and Capital* (1939): hedgers (net short) pay speculators (net long) a premium to
  bear price risk. Paul Cootner, *"Returns to Speculators: Telser versus Keynes"*, **Journal of Political
  Economy** 1960, formalised hedging pressure.
- **The modern cross-sectional evidence.** Gary Gorton, Fumio Hayashi & K. Geert Rouwenhorst, *"The
  Fundamentals of Commodity Futures Returns"*, **Review of Finance** 17(1), 2013, and Devraj Basu & Joëlle
  Miffre, *"Capturing the Risk Premium of Commodity Futures"*, **Journal of Banking & Finance** 2013: net
  hedger/speculator positioning and inventory predict the cross-section of commodity returns.

## The honest counters — why the verdict is `WEAK` / `MIRAGE` / `Faded`

- **Post-publication decay / crowding.** A premium readable from a free public report (the CFTC COT) is
  exactly the kind that gets competed away once commodity risk-premia funds and CTAs all trade it
  (McLean & Pontiff, *Journal of Finance* 2016, on the general decay). The factor's negative Sharpe across
  the modern sample is consistent with that.
- **Sample/breadth dependence.** The academic results use deep histories and broad cross-sections; this
  study's liquid-12, ~10-year (2015–2025) window is thinner, and the legacy COT's commercial bucket mixes
  true hedgers with swap dealers — a stated limitation, and the first beat-7 fork (disaggregated COT,
  more contracts, longer history).

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference** (Newey & West, *Econometrica* 1987) on the long-short factor.
- **Data.** Real run uses **CFTC legacy futures-only Commitments of Traders** (free, `deacot<YYYY>.zip`
  from cftc.gov) for commercial positioning, and **Yahoo continuous front-month commodity futures**;
  weekly COT lagged and forward-filled onto the return grid (`data.align_hp`, causal). Pinned with
  [`quantlab.repro`](../../../quantlab/repro.py).

## Caveats stated in the open (house rule)

- **Legacy COT, commercial = hedger proxy.** The legacy report's "commercial" bucket includes swap
  dealers, not only physical hedgers; the disaggregated report separates them (a fork).
- **Continuous front-month futures, weekly, USD.** A simplification of the roll-adjusted return; the
  hedging signal is slow, so weekly is the right horizon, stated not hidden.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
