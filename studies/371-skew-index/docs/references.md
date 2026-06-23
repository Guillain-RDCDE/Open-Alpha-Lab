# References & literature map — Study 371 (CBOE SKEW Index)

## The claim under test

- **The index (CBOE).** Chicago Board Options Exchange, *The CBOE SKEW Index — SKEW* (white
  paper, 2010/2011). SKEW is computed from the prices of out-of-the-money S&P 500 options and is
  scaled so that **100 = a normal (symmetric) risk-neutral return distribution**; readings of
  115–145 correspond to progressively fatter left (crash) tails. The marketing line is that SKEW
  measures **tail risk** — the perceived probability of an outlier two-or-three-standard-deviation
  move — that the at-the-money **VIX** does not capture.
- **The folklore.** SKEW is routinely framed in financial media as a "black-swan index" or
  "crash gauge" — *"when SKEW spikes, the smart money is buying crash insurance, so watch out."*
  The implied claim is **incremental**: SKEW warns precisely when VIX is calm, so a high SKEW
  should precede weak forward returns and/or elevated tail events even after conditioning on VIX.

## Why "informative beyond VIX" is the right question

- **VIX and the variance risk premium.** Both SKEW and VIX are *risk-neutral* moments of the
  S&P 500 return distribution priced from options (Bakshi, Kapadia & Madan, 2003, *Stock Return
  Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options*, RFS,
  give the model-free formulas for implied variance and skewness). Because the second and third
  moments co-move, the empirical question is never "does SKEW correlate with anything" but
  "**does SKEW add information once VIX is in the regression**." We answer it with a forward-return
  regression on standardised SKEW *and* VIX.
- **Does implied skew predict returns?** The academic record is, at best, mixed and largely
  about the *cross-section* of single stocks, not the index tail. Conrad, Dittmar & Ghysels
  (2013, *Ex Ante Skewness and Expected Stock Returns*, Journal of Finance) and Bali & Murray
  (2013) find option-implied skewness relates to the cross-section of equity returns; that is a
  different object from "index SKEW times-series-predicts the S&P." Studies that test the CBOE
  SKEW index directly (e.g. Bevilacqua & Tunaru, 2021, *The SKEW index: extracting what has been
  left*, and practitioner notes from CBOE and Bloomberg) generally find **weak or no**
  out-of-sample timing value once VIX/level effects are controlled.

## Why the naive bucket *t* is a trap — the statistics

- **Overlapping windows & pseudo-replication.** Forward 1/3-month returns measured daily are
  enormously autocorrelated; treating each day as an independent draw inflates *t*-stats
  dramatically. The fix is HAC standard errors (Newey & West, 1987, *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica) and/or a **block bootstrap** that resamples contiguous blocks (Künsch, 1989;
  Politis & Romano, 1994, the stationary/circular bootstrap). High-SKEW days cluster into ~a
  dozen *episodes*, so the effective sample is tiny — a point made generally by Welch (1947) on
  unequal-variance two-sample tests and by the small-sample / multiple-testing critiques of
  Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected Returns*, RFS).
- **Base rates and the warning illusion.** US equities rise in most months and crash in few, so
  *any* "warning" indicator is graded against a low unconditional tail-event base rate; a high
  conditional win-rate is expected under the null (the base-rate fallacy, Kahneman & Tversky,
  1973, *On the psychology of prediction*). We therefore score the **tail rate after high SKEW
  against the unconditional tail rate**, with a placebo null.

## Method lineage (the desk's shared engine)

- **VIX-controlled HAC regression.** [`strategy.regress_forward`](../skew_index/strategy.py) —
  the decisive Signal-axis test: forward return on standardised SKEW *and* VIX with Newey-West
  *t*-stats.
- **Block bootstrap of the bucket gap.**
  [`strategy.block_bootstrap_decile`](../skew_index/strategy.py) — the autocorrelation-respecting
  *t* for the high-SKEW-vs-rest difference, plus the count of distinct high-SKEW episodes.
- **Tail-warning + placebo null.** [`strategy.tail_warning`](../skew_index/strategy.py) and
  [`strategy.placebo_pvalue`](../skew_index/strategy.py) — forward tail-event rate after high
  SKEW vs base rate, with a randomization null sized to the bucket.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../skew_index/data.py) plants a
  *VIX-orthogonal* SKEW edge via a knob; with the knob off the inference must stay quiet, with it
  on it must light up. The offline core runs with no network.
- **Forward-return measurement with execution lag.**
  [`strategy.forward_return`](../skew_index/strategy.py) enters one day after the signal (no
  look-ahead); the costed sleeve in [`strategy.fade_sleeve`](../skew_index/strategy.py) applies a
  one-way cost per position change.

## Data sources used here

- **yfinance** daily closes for `^SKEW`, `^VIX` and `SPY`, joint window 1993-01-29 → 2026-06-18,
  cached under `_cache/skew_vix_spy.csv`. SPY's 1993 inception sets the start. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py). SPY is a **price-only** proxy here (no
  dividends), labelled on the Tradability axis.

## Related desk studies

- **VIX-term / volatility-carry studies** in the *Options & volatility* family — the SKEW
  question is the natural "does the *third* moment add anything to the *second*" follow-up to
  whether implied vol itself times the market.
