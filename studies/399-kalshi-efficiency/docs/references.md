# References & literature map — Study 399 (Kalshi-Efficiency)

## The claim under test

- **Kalshi & event contracts.** Kalshi is a CFTC-regulated designated contract market for
  **binary event contracts**: each market pays $1 if an event resolves YES and $0 otherwise, so
  the price (in cents) is a direct, tradable probability. The believer's pitch is *price
  efficiency vs edge*: if the crowd systematically misprices some bucket of contracts, the gap
  between price and realized frequency is a free edge — buy the cheap longshots short / the rich
  favourites long.
- **Prediction markets as probabilities.** Wolfers & Zitzewitz (2004), *Prediction Markets*
  (Journal of Economic Perspectives), and Arrow et al. (2008), *The Promise of Prediction
  Markets* (Science): contract prices aggregate information into well-calibrated probability
  forecasts — the efficient-markets case for taking the price at face value.

## The favourite–longshot bias — the documented distortion

- **The racetrack original.** Griffith (1949) and Ali (1977), *Probability and Utility
  Estimates for Racetrack Bettors* (Journal of Political Economy): bettors **over-bet
  longshots** and **under-bet favourites**, so longshots win *less* often than their odds imply
  and favourites *more* often — a systematic, monotone miscalibration. Thaler & Ziemba (1988),
  *Parimutuel Betting Markets* (JEP), is the canonical survey.
- **In prediction/event markets.** Page & Clemen (2013) and Snowberg & Wolfers (2010),
  *Explaining the Favorite–Longshot Bias* (Journal of Political Economy): the same tilt appears
  in binary prediction markets, but it is **small** and shrinks toward the center; whether it is
  *harvestable* after transaction costs is the open question this study's machinery targets.

## Why a calibration curve from a finite book is slippery — the statistics

- **Calibration / reliability.** Murphy (1973), *A New Vector Partition of the Probability
  Score* (Journal of Applied Meteorology): the **Brier-score decomposition** into
  *reliability + resolution − uncertainty*. The reliability term is the calibration error we
  read directly; a finite book makes it wiggle even under perfect calibration. DeGroot &
  Fienberg (1983), *The Comparison and Evaluation of Forecasters*, formalise calibration and
  refinement.
- **Small/finite-sample inference.** With per-bucket counts in the hundreds, an empirical
  frequency has standard error √(p(1−p)/n); a few-cent gap can be noise. We test the harvestable
  spread with a **Welch/Student t** (Welch, 1947) against zero and, because the right null is
  *"a calibrated book,"* a **within-bucket randomization test** (Fisher's randomization logic;
  Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993) — resample YES with probability
  equal to bucket price and ask how often chance reproduces the spread.
- **Base-rate vs edge.** A long-favourite book has a **high win-rate by construction** —
  favourites win often — which says nothing about an edge; the edge is the *calibration gap net
  of fees*, not the win-rate. The base-rate fallacy (Kahneman & Tversky, 1973, *On the
  Psychology of Prediction*) is exactly the trap the "92% win-rate!" pitch sets.

## Why there is no free real Kalshi tape — and what we do instead

- **No free resolved-history feed.** Kalshi's full resolved-contract history (prices +
  settlements) is **not** available for free; the public API is rate-limited and live-oriented.
  We therefore build a **transparent, clearly-labelled illustrative book** — a seeded set of
  binary contracts whose prices are calibrated *except* for a tunable favourite–longshot tilt —
  and treat the whole study as a **methods demo**, not a backtest. Every input is constructed
  and named as such; the significance on the illustrative tape is *planted* and never backs a
  `REAL` Signal stamp.

## Method lineage (the desk's shared engine)

- **Calibration curve + Brier decomposition.**
  [`strategy.calibration_curve`](../kalshi_efficiency/strategy.py) and
  [`strategy.brier_decomposition`](../kalshi_efficiency/strategy.py) — the reliability table and
  Murphy's miscalibration term.
- **Spread + Welch t + randomization null.**
  [`strategy.contract_pnl`](../kalshi_efficiency/strategy.py),
  [`strategy.welch_t`](../kalshi_efficiency/strategy.py) and
  [`strategy.randomization_pvalue`](../kalshi_efficiency/strategy.py) — the harvestable
  long-favourite/short-longshot spread, net of a one-way fee, with a within-bucket calibrated
  null.
- **Deterministic synthetic control.**
  [`data.synthetic_book`](../kalshi_efficiency/data.py) plants a known favourite–longshot edge;
  the offline core runs with no network. The control confirms the engine is faithful *and* that
  `edge=0` stays quiet (no false positive) while the fee turns a fair book into a pure cost.

## Data sources used here

- **Illustrative resolved-contract book** built by
  [`data.fetch_real`](../kalshi_efficiency/data.py), cached under
  `_cache/kalshi_contracts.csv` (price, outcome, decorative resolution stamp). All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 351 — BTC 5-minute Polymarket momentum](../351-btc-5m-polymarket-momentum/)**: a
  prediction-market sibling — a genuinely high win-rate that is entirely priced in, so the
  favoured side is quoted at its own win-rate and the edge is zero net of the spread. Same
  "high-confidence number, no harvestable edge" shape, on a different exchange.
- **[Study 346 — Multiple-Testing](../346-multiple-testing/)** and the research-method demos
  343–350: methods demos on synthetic nulls with *provably nothing to find*; Study 399 is the
  prediction-market member of that family — a transparent illustration of the calibration
  machinery, not a claim of a new real edge.
