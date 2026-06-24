# References & literature map — Study 443 (Volume Profile POC)

## The claim under test

- **The folklore.** "The **Point of Control** (POC) — the price level with the most traded
  volume in a session — is 'fair value'. Price is drawn back to it like a **magnet**: if the next
  day opens away from the POC, fade toward it and target the level; the POC gets *respected*." It
  is the central claim of **Volume Profile / Market Profile** day-trading.
- **The source framework.** J. Peter Steidlmayer's **Market Profile** (developed at the CBOT in
  the 1980s; Steidlmayer & Koy, *Markets and Market Logic*, 1986; Steidlmayer & Hawkins, *Steidlmayer
  on Markets: Trading with Market Profile*, 2003) introduced the POC, the Value Area, and the TPO
  (time-price-opportunity) profile. The volume-profile variant (volume-at-price instead of
  time-at-price) is the modern retail/futures-desk staple — popularised by writers such as Jim
  Dalton (*Mind Over Markets*, 1990) and a generation of order-flow educators.
- **Why it is testable folklore.** The POC is a precise, observable level computed from public
  data, and the claim ("price returns to it") makes a sharp, falsifiable prediction about the next
  session's range — exactly the kind of support/resistance assertion the desk audits.

## The honesty trap — the distance-matched control

- **The support/resistance fallacy.** On a volatile intraday tape, the high-low range covers a
  few percent, so *any* level near the open is touched with high probability — independent of
  whether it "means" anything. Claiming a level "works" because it is often touched is the same
  error as claiming a wide net catches a particular fish. The estimand must therefore be a
  **difference**: POC touch-rate minus the touch-rate of a level placed the *same distance* from
  the open. This control logic mirrors the placebo discipline throughout the desk
  (round-numbers, data-mining-roulette, multiple-testing demos).
- **No-look-ahead.** The POC of session $d-1$ is known at the prior close, so it is a legitimate
  predictor of session $d$'s range — no information leaks forward (standard event-study timing).

## Inference

- **Paired one-sample / HAC *t*.** The per-event (POC-touch − control-touch) indicator is tested
  against zero with a one-sample *t* and a **Newey-West (HAC)** standard error (Newey & West,
  1987, *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*, Econometrica) for the day-over-day autocorrelation of touch outcomes
  (volatility clusters). The **t ≥ 2** bar is the desk's REAL gate.
- **Wilson intervals.** Touch-rates carry Wilson score intervals (Wilson, 1927, *Probable
  Inference, the Law of Succession, and Statistical Inference*, JASA) — the honest small-sample
  interval for a proportion, used for every conditional rate per METHODOLOGY.
- **Shuffle-the-POC placebo.** Re-pair each session's range with a permuted POC offset thousands
  of times (Fisher's randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*,
  1993) and ask whether the *true* session-to-its-own-POC pairing is touched more than a random
  pairing. It severs only the bond the magnet claim depends on.

## Microstructure & capacity

- **Intraday levels die to the spread.** The fade pays the bid-ask spread on entry and exit; for
  intraday strategies this is the binding constraint. Even granting a thin edge, the break-even
  cost analysis (here **negative** — no gross edge at all) is the capacity verdict. Background:
  Roll (1984, *A Simple Implicit Measure of the Effective Bid-Ask Spread*, JF) on the spread's
  role in intraday returns; Frazzini, Israel & Moskowitz (2018, *Trading Costs*) on paper-vs-net.
- **Short-span data.** Yahoo caps 5-minute US-equity history at ~60 calendar days, so this is a
  deliberately short tape — adequate to reject a strong magnet on liquid names, underpowered for a
  tiny conditional edge (named loudly in `docs/results.md`).

## Method lineage (the desk's shared engine)

- **POC / volume profile.** [`data.session_poc`](../volume_profile_poc/data.py) — highest-volume
  bin of the session's typical-price volume profile.
- **Distance-matched control + touch test.** [`strategy.control_level`](../volume_profile_poc/strategy.py),
  [`strategy.reversion_stats`](../volume_profile_poc/strategy.py) — the POC vs random-level
  touch-rate difference with paired & HAC *t* and Wilson intervals.
- **Shuffle-the-POC placebo.** [`strategy.placebo_pvalue`](../volume_profile_poc/strategy.py).
- **Deterministic synthetic control.** [`data.synthetic_panel`](../volume_profile_poc/data.py)
  plants a known pull toward the POC; with the pull set to zero the detector must NOT manufacture
  a magnet — the offline core runs with no network.

## Data sources used here

- **yfinance** 5-minute RTH bars (auto-adjusted) for SPY, QQQ, AAPL, MSFT, NVDA, TSLA,
  2026-03-31 → 2026-06-23, cached under `_cache/vp_*_5m.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[../116-power-hour](../116-power-hour)** — the desk's other intraday level/timing teardown
  (last-hour continuation), same session-panel idiom and short-span discipline.
- **[../376-moc-imbalance](../376-moc-imbalance)** and **[../377-bid-ask-bounce](../377-bid-ask-bounce)**
  — intraday microstructure studies where the spread is the binding constraint, as here.
- The **research-method demos** (round-numbers, data-mining-roulette, multiple-testing) frame why
  a level must beat a **distance-matched random control**, not zero — the POC is the textbook case
  where the naive "it gets touched a lot!" reading dissolves under the right control.
