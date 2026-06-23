# References & literature map — Study 379 (ETF Lead-Lag)

## The claim under test

- **The folklore.** A staple of trading-desk lore and microstructure commentary: the big,
  liquid ETF (here **SPY**) is where price discovery happens first, so its smaller, slower,
  less-liquid members "catch up a day later." If a move in the leader genuinely *predicts* the
  next move in the laggards, you could buy the slow members the instant the leader pops — a
  clean mechanical edge. We test the strongest tradable form: does *yesterday's* leader return
  predict *today's* member return, net of costs?
- **Why it sounds right.** Liquid instruments do incorporate information faster; the literature
  on **lead-lag** is real. The question is the *scale*: genuine ETF/constituent lead-lag is an
  **intraday / tick** phenomenon, and the daily-bar version most retail framings imply is a
  much weaker claim that has to survive realistic costs.

## The lead-lag literature (the real version, mostly intraday)

- **Lo & MacKinlay (1990), *When Are Contrarian Profits Due to Stock Market
  Overreaction?*** (Review of Financial Studies). The classic lead-lag result: large stocks
  lead small stocks in *weekly* returns, and cross-autocovariances drive a large share of
  short-horizon "contrarian" profits — but the effect is small and contaminated by
  microstructure.
- **Hou (2007), *Industry Information Diffusion and the Lead-Lag Effect in Stock Returns***
  (RFS). Big firms lead small firms *within* industries; the lead-lag is concentrated and
  largely an information-diffusion story, not a free lunch after costs.
- **Chordia, Sarkar & Subrahmanyam (2011)** and the broader **price-discovery / information
  share** literature (Hasbrouck, 1995, *One Security, Many Markets*; Gonzalo & Granger, 1995).
  Price discovery concentrates in the most liquid venue; the laggards' "catch-up" is the slow
  side of the same information event, measured in **minutes**, not days.
- **ETF-specific lead-lag.** Work on SPDR/ETF arbitrage and index-futures-vs-cash lead-lag
  (e.g. Stoll & Whaley, 1990, *The Dynamics of Stock Index and Stock Index Futures Returns*)
  finds the index/ETF leads the cash basket — again **intraday**, arbitraged away within the
  trading day by authorized participants and index arbitrageurs.

## Why daily bars are the wrong (and decisive) clock — and what we do

- **The contemporaneous trap.** On daily bars the leader and the members co-move enormously
  *the same day* (here **corr ≈ 0.87** at lag 0). That co-movement is **untradable**: you
  cannot act on a same-day move you'd have to know in advance. The only tradable object is the
  **next-day** cross-correlation (leader\_{t−1} → members\_t), which is what we isolate.
- **Microstructure contamination of measured lead-lag.** Non-synchronous trading and bid-ask
  bounce **manufacture** spurious lead-lag and autocorrelation (Scholes & Williams, 1977;
  Lo & MacKinlay, 1990; Roll, 1984, *A Simple Implicit Measure of the Effective Bid-Ask
  Spread*). A naive daily cross-correlation can show a "lead" that is pure stale-pricing
  artefact, not exploitable information — so a positive bump alone is not enough; it must
  survive a HAC test and, crucially, **costs**.
- **HAC inference + a placebo null.** We test the one-day lead-lag slope with a Newey-West HAC
  *t* (Newey & West, 1987) — overlapping/autocorrelated daily data inflate a naive *t* — and
  the next-bar rule with a **placebo / randomization** null sized to the trade count (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).

## Method lineage (the desk's shared engine)

- **Cross-correlation profile + HAC slope.**
  [`strategy.crosscorr_profile`](../etf_lead_lag/strategy.py) and
  [`strategy.lag1_beta`](../etf_lead_lag/strategy.py) — the lead-lag profile and a
  Newey-West-robust one-day slope (the Signal-axis tests).
- **Next-bar rule + placebo null.** [`strategy.summarize`](../etf_lead_lag/strategy.py),
  [`strategy.placebo_pvalue`](../etf_lead_lag/strategy.py) and
  [`strategy.net_of_costs`](../etf_lead_lag/strategy.py) — conditional next-day member return
  vs the unconditional base rate, a 20,000-draw randomization null, and one-way costs × turnover.
- **Deterministic synthetic control.**
  [`data.synthetic_leadlag`](../etf_lead_lag/data.py) plants a **known** one-day lead via a
  single knob; the offline core runs with no network. The control confirms the engine **would**
  detect a real lead (edge=0 stays quiet, a planted lead lights up) — a machinery proof, never
  market evidence.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + a fixed 37-name smaller/less-liquid US basket,
  2000-01-04 → 2026-06-18, cached under `_cache/lead_lag_prices.csv`. All headline numbers are
  pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 140 — Amihud-Illiquidity](../140-amihud-illiquidity/)**: the liquidity premium that
  this lead-lag story leans on — whether illiquidity itself is a priced, tradable factor.
- **The microstructure family generally**: lead-lag, illiquidity and turnover effects are real
  *in the literature* and overwhelmingly *intraday*; the recurring desk lesson is that they
  evaporate once you move to daily bars and charge realistic costs.
