# References & literature map — Study 355 (Magnificent-Seven)

## The claim under test

- **The "Magnificent Seven" label.** Coined by Bank of America's Michael Hartnett in 2023 (a
  nod to the 1960 film), it bundles AAPL, MSFT, GOOGL/GOOG, AMZN, NVDA, META and TSLA — the seven
  US mega-caps that drove the bulk of the S&P 500's 2023-2024 gains. The popular claim, repeated
  across financial media and retail forums: *"hold the Mag 7 equal-weight and you crush the index."*
  The testable claims are (H₁) the basket genuinely out-earns the S&P 500 on the tape; (H₂) that
  spread is a **forward-tradable factor** you could have captured in advance; (H₃) "hold the
  Mag 7" is a repeatable strategy rather than a label for the realised winners.
- **The catch the label hides.** The seven are *named because they won*. The basket's membership
  is defined with the full 2015-2026 sample in view — a textbook **look-ahead** (and, since the
  field excludes mega-caps that died, a **survivorship**) selection. The study's job is to
  separate the real recent spread (H₁, true) from the forward edge (H₂/H₃, the trap).

## Why selection-after-the-fact manufactures a spread

- **Survivorship bias.** Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias in
  Performance Studies* (Review of Financial Studies); Elton, Gruber & Blake (1996),
  *Survivorship Bias and Mutual Fund Performance*. Selecting a sample on its end-state inflates
  measured returns because the failures were removed before the test saw them. The desk's
  **[Study 345 — Survivorship-Bias](../../345-survivorship-bias/)** plants and measures exactly
  this; this study is its single-basket cousin.
- **Look-ahead / data-snooping.** Lo & MacKinlay (1990), *Data-Snooping Biases in Tests of
  Financial Asset Pricing Models*; White (2000), *A Reality Check for Data Snooping*. Choosing the
  basket *after* observing the outcomes is the strongest form of snooping: the spread is a
  guaranteed artefact, quantified here by the ex-post "pick the winners" placebo on a no-edge
  synthetic tape.
- **The Malkiel dartboard / random portfolios.** Malkiel (1973), *A Random Walk Down Wall Street*;
  the desk's **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)**. A *blindly*
  chosen equal-weight basket is the honest forward benchmark; where the hand-picked Mag 7 sits in
  that random distribution (here, the 99.7th percentile) measures how much is selection.

## Why concentration ≠ a factor

- **Equal-weight vs cap-weight.** Plyakha, Uppal & Vilkov (2012), *Why Does an Equal-Weighted
  Portfolio Outperform Value- and Price-Weighted Portfolios?*. Equal-weighting tilts toward
  smaller, higher-return names; we strip this with an **equal-weight field** control so the
  residual Mag-7 spread is *name selection*, not just weighting.
- **Cross-sectional momentum & its decay.** Jegadeesh & Titman (1993), *Returns to Buying Winners
  and Selling Losers*; the well-documented post-publication decay of factor premia. "Yesterday's
  winners keep winning" is a real but mean-reverting, regime-dependent effect — not the permanent
  free lunch the Mag-7 label implies.
- **Index concentration.** S&P Dow Jones Indices research on the rising weight of the largest
  names in the S&P 500 (the Mag 7 reached ~30%+ of the index by 2024). A cap-weighted index
  *already owns* the winners; the marginal "edge" of the equal-weight Mag-7 basket over SPY is a
  leveraged bet on continued mega-cap dominance, not a diversifying factor.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*. The Signal-axis test is
  the HAC t-stat of the monthly basket-minus-benchmark spread (`strategy.hac_tstat_diff`).
- **One-way turnover costs on NAV.** Costs charged as `Σ|Δw| × cost_bps` per rebalance on the
  drifted book (`strategy._book_returns`), mirroring the desk's other equity studies. For a
  7-name monthly basket they are immaterial — the mirage is selection, not friction.
- **Deterministic synthetic control.** A fixed-seed monthly panel
  ([`data.synthetic_panel`](../magnificent_seven/data.py)) with a tunable pre-named
  ``alpha_spread`` and an ex-post placebo ([`strategy.expost_winners`](../magnificent_seven/strategy.py))
  — the offline core runs with no network and reproduces for any reader.

## Data sources used here

- **yfinance** monthly total-return closes (auto-adjusted), 2015-08 → 2026-05, mega-cap field +
  SPY, cached under `_cache/`. All headline numbers are pinned in [`results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 345 — Survivorship-Bias](../../345-survivorship-bias/)**: the panel-level version —
  delisting deleted before the backtest sees it.
- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)**: the random-basket
  benchmark and the equal-vs-cap-weight decomposition reused here.
- **[Study 347 — Look-Ahead Bias](../../347-look-ahead-bias/)** and
  **[Study 346 — Multiple-Testing](../../346-multiple-testing/)**: the machinery of why selecting
  on the outcome guarantees a spurious "edge."
