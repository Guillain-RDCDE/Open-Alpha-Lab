# References & literature map — Study 947 (The Buffer Ladder)

## The claim under test

- **The laddering pitch.** A defined-outcome (buffer) ETF gives you a stated buffer against
  the underlying's losses and a stated cap on its gains, over a **fixed outcome period**
  that resets annually on a named month. That makes the fund's payoff depend on *when you
  bought it* — where the underlying sits relative to that vintage's reset strike. Buy PJAN
  in November and you are holding a very different option position from the one the
  January buyer holds. The industry's answer is the **laddered wrapper**: one ticker
  (Innovator's **BUFR**, the Laddered Allocation Power Buffer ETF) holding a spread of
  vintages, so entry timing "averages out". It charges a management fee **on top of** the
  acquired funds' expense ratios.
- **The obvious retort, and this study's question.** Nothing stops a private investor from
  buying four vintages himself and equal-weighting them — four trades and a rebalance
  reminder. So: does the wrapper deliver a **laddering premium** beyond that home-made
  basket, and is the entry-point luck it averages away even large enough to be worth a fee
  layer? We test both on the excess-of-cash, beta-matched, costed tape.

## Why the answer could plausibly be yes

- **Path dependency in defined-outcome payoffs.** Because a buffer fund's terms are struck
  once a year, its delta, its distance to the cap and its distance to the buffer all drift
  through the period. Bhansali & Harris (2018), *Everybody's Doing It: Short Volatility
  Strategies and Shadow Financial Insurers* (Financial Analysts Journal), and the
  option-overlay literature generally, document how much a collar's realised payoff depends
  on where it was struck relative to spot. A ladder is the obvious structural fix.
- **Diversifying an idiosyncratic timing draw.** The classic case for spreading a purchase
  across dates. Constantinides (1979), *A Note on the Suboptimality of Dollar-Cost
  Averaging as an Investment Policy* (Journal of Financial and Quantitative Analysis) is the
  sceptical pole; the practitioner case is that averaging removes a variance you were never
  compensated for. Which side wins depends entirely on **how correlated the tranches are** —
  which is precisely what this study measures.
- **A fund-of-funds can add value when the constituents are hard to hold.** Elton, Gruber &
  Blake (2007), *Participant Reaction and the Performance of Funds Offered by 401(k) Plans*
  (Journal of Financial Intermediation) — wrappers earn their keep through behaviour and
  operations, not returns. A real effect, and not one a return series can price.

## Why the answer is likely no

- **Fees compound, laddering does not.** The wrapper's incremental fee is certain; the
  laddering benefit is a variance reduction whose size falls with the correlation between
  vintages. Sharpe (1991), *The Arithmetic of Active Management* (Financial Analysts
  Journal) is the general form of the argument; our measured 0.889 pairwise correlation is
  the specific one.
- **Beta masquerading as skill.** Any wrapper holding a different mix of strikes and
  maturities than the four quarterly vintages will carry a different equity beta, and beta
  is not a premium — Jensen (1968), *The Performance of Mutual Funds in the Period
  1945-1964* (Journal of Finance), is where the desk's beta-matching discipline comes from.
  On this tape BUFR's SPY-beta is 0.579 against the four-vintage basket's 0.439, which
  accounts for the whole of its apparent outperformance.
- **The defined-outcome category has already been shown to be fairly priced.** Our Study 624
  found the buffer funds' 5-8 pp/yr shortfall vs SPY total return collapses to ~0 against a
  beta-matched SPY/BIL mix — comfort priced correctly, neither free lunch nor rip-off. If
  the constituents are fairly priced, a wrapper that merely holds several of them has very
  little room to add anything.

## Related desk studies (dedup)

- **[Study 624 — Buffer-ETF-Cost](../../624-buffer-etf-cost/)**: the direct parent, and the
  one to read first. It asks what **one** buffer fund costs you — mechanical delivery of the
  stated cap and buffer, the shortfall vs SPY total return, and the race against a
  beta-matched SPY/BIL mix. Study 947 takes that as settled and asks the **next** question,
  one layer up the wrapper stack: given that a single vintage is fairly priced, does
  *bundling* the vintages add anything? Different comparison set (wrapper vs its own
  constituents, not fund vs market), different measurement (a laddering premium and a
  correlation-driven variance reduction, not a cap/buffer delivery check).
- **[Study 921 — Bill-Ladder-vs-ETF](../../921-bill-ladder-vs-etf/)**: the identical
  question on the *cash* shelf — a home-made T-bill ladder against the cash ETF, where the
  DIY margin turns out to be exactly the expense ratio and nothing more. Study 947 is the
  same DIY-vs-wrapper frame on a *derivative* product, where the wrapper's extra layer is
  smaller but the constituents are far more complex.
- **[Study 892 — Corporate-Bond-Ladder](../../892-corporate-bond-ladder/)**: a bond ladder
  against a bond ETF once **duration** is matched — the same "match the exposure before you
  compare" discipline that this study applies with **beta**.
- **[Study 934 — Lump-Sum-vs-DCA](../../934-lump-sum-vs-dca/)** and
  **[Study 937 — Tranched-Rebalancing](../../937-tranched-rebalancing/)**: spreading a
  purchase or a rebalance across dates, where the apparent benefit turns out to be an
  exposure difference and vanishes once matched. Study 947 finds the *same* pattern in a
  place the industry sells it as a product feature — but measures the mechanism directly
  (the vintage correlation) rather than only the outcome.
- **[Study 337 — Covered-Call-ETF](../../337-covered-call-etf/)**: the other option-wrapper
  family — sell the upside for a distribution. Orthogonal payoff, same methodological spine.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)** and
  **[Study 936 — Rebalance-Bands](../../936-rebalance-bands/)**: the rebalancing-premium
  literature this study's DIY basket leans on when it resets to equal weights.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.newey_west_t`](../buffer_ladder/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) test.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures* (Journal of Finance), in its
  Newey-West form — [`strategy.gap_tstat`](../buffer_ladder/strategy.py).
- **Circular block bootstrap, paired across arms.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA) — [`strategy.bootstrap_gap_ci`](../buffer_ladder/strategy.py) and
  [`strategy.bootstrap_sharpe_gap_ci`](../buffer_ladder/strategy.py), which resample both
  arms on the *same* block indices so the 0.960 correlation between them is preserved.
- **Beta matching before comparison.** Jensen (1968), as above;
  [`strategy.expanding_beta`](../buffer_ladder/strategy.py) estimates it out-of-sample on an
  expanding window and lags it one day, so the matched arm is a portfolio you could have
  actually held.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — as-of slice
  plus content fingerprint on every headline run.

## Data sources and declared non-tape inputs

- **BUFR** (Innovator Laddered Allocation Power Buffer ETF), **PJAN / PAPR / PJUL / POCT**
  (the four quarterly Innovator S&P 500 Power Buffer vintages), **SPY** (the underlying),
  **BIL** (1-3 month T-bill, the cash leg) — daily **total-return** closes via `yfinance`
  (`auto_adjust=True`), 2020-08-11 → 2026-06-30. TR vs TR throughout: the vintages make
  annual capital-gain distributions and BIL's entire return *is* a distribution, so a
  price-only series would mis-rank every arm. (Note that a buffer fund's *stated terms* are
  written on the underlying's **price** return — that is Study 624's measurement, not this
  one; here every comparison is fund-total-return against fund-total-return.)
- **PROXY / ASSUMPTION — the fee layer.** `FEE_SINGLE_VINTAGE_PCT = 0.79`%/yr (quoted
  prospectus expense ratio of one Power Buffer vintage) and `FEE_LADDER_EXTRA_PCT = 0.20`%/yr
  (the wrapper's assumed incremental layer over the acquired funds' fees). Neither is
  measured on the tape — published NAV returns are already net of whatever was charged — so
  both are used **only** to build a "had the layer been waived" counterfactual, and
  `FEE_EXTRA_GRID_PCT` sweeps the assumption from 0.00 to 0.40%/yr. No conclusion rests on
  the guess.
- **PROXY — the DIY ladder's composition.** BUFR's mandate is a ladder of Power Buffer
  vintages; our home-made comparator is the four **quarterly** vintages with a usable
  history from before BUFR's inception. It is therefore a *sparse* ladder, not a
  constituent-exact replication — which is why the two series track at 0.960 rather than
  0.999, and why the beta match is necessary rather than optional.
- **Survivorship, named.** BUFR and these four vintages are the *surviving* flagships of a
  category that has launched and closed products. The panel flatters the funds. More
  importantly there is exactly **one** laddered wrapper with this history: this is an n-of-1
  product test over 5.9 years containing a single genuine down-year (2022), not a
  cross-sectional result.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps
  between reruns; the common window is gated by BUFR's 2020-08-11 inception.
