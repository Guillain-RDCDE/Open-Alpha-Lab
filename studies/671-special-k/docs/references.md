# References & literature map — Study 671 (Special K)

## The claim's source

- **Pring, Martin J.** *Momentum Explained, Volume 2* (2004) and the **StockCharts
  ChartSchool** entry "Special K" — the canonical daily parameters used here: twelve
  SMA-smoothed rate-of-change series with lookbacks **(10, 15, 20, 30, 50, 65, 75, 100, 195,
  265, 390, 530)**, weighted **1-2-3-4** within four bands, summed, crossed against a 100-day
  signal SMA. Pring designed Special K explicitly as a **"reduced-whipsaw"** cousin of his own
  Know Sure Thing (KST) — folding in enough scales (two weeks to two years) to catch what he
  calls **"major reversals"** or "primary trend changes" in one line, so the crossover itself
  is sold as a cyclic-turn signal, not just a trend filter.
- **StockCharts ChartSchool — "Know Sure Thing (KST)"** for the parent indicator's own
  documentation and marketing language ("Know Sure Thing" — the branding is explicit); Special
  K is presented as KST's big sibling, summing three times as many ROC series.
- **Investopedia** and assorted charting-platform glossaries repeat the "flags major market
  turning points, filtered of whipsaw" framing verbatim — the popular form of the claim we
  steelman and test.

## What we measure, and the honesty rails

- **Event study.** Post-crossover daily returns vs baseline, Newey-West HAC *t* with
  **lags = horizon** — the window length is exactly the autocorrelation this design induces
  (overlapping forward windows), so the lag choice isn't tuned for a flattering answer; it's
  set by construction (same idiom as study 637's event-window Newey-West cross-check).
- **Random-timing placebo**, Coppock-style (see [105-coppock-curve](../105-coppock-curve/)):
  draw N random dates of the same count as the real crossovers, same horizon, and ask how
  often a random calendar beats the observed mean — a nonparametric answer that doesn't
  assume independence across overlapping forward-return windows.
- **Long/flat timer.** NET excess-of-cash Sharpe raced against buy-and-hold and a one-line
  200-day SMA (Faber), with a Newey-West HAC *t* on the daily excess returns, a sign-flip
  permutation placebo on the exposure schedule, a cost sweep, and **one documented execution
  lag** (signal known at close *t*, position earns the return of *t+1* — a single `shift`).
- **Three independent real tapes**, not one: SPY daily (total-return, 1993-2026, the desk's
  house-standard tape), ^GSPC daily (**price-only, no dividends** — named everywhere it
  appears — 1962-2026, 64 years and every major post-war cyclic turn), and SPY resampled to
  weekly bars with periods scaled by /5. A claim about "major cyclic turns" that only shows up
  on one tape, one bar frequency, is not a claim that survived — it's a claim that got lucky
  once.
- **Parameter robustness.** Every ROC/SMA period scaled by a common factor (0.7×/1.0×/1.3×,
  weight structure and band shape held fixed) — asks whether Pring's exact numbers are special
  or any similarly-shaped multi-scale blend performs about as well (or as poorly).

## Why the synthetic control uses a regime cycle, not an AR(1) trend

- Sibling momentum-oscillator studies on this desk (KST, ROC, DPO) plant a one-day
  autocorrelation coefficient in their synthetic tape, because their crossovers respond to
  lookbacks of a few weeks. Special K's slowest component is a **530-day** ROC smoothed by a
  **530-day** SMA — a one-day AR(1) coefficient decays to nothing over that horizon and cannot
  power-test the part of the indicator that is actually distinctive. We instead plant a
  **two-state Markov bull/bear regime** with ~4.6-year average sojourns (matched to Special
  K's own longest lookback) and a tunable drift differential — the honest positive control for
  a "major cyclic turn" claim, and the null (`amp=0`) is checked never to manufacture a
  false-positive edge across 20 seeds.

## Data sources

- **SPY daily total-return closes** and **^GSPC daily price-only closes** — yfinance (no
  key), cached under `_cache/` (`sk_spy.csv`, `sk_gspc.csv`), through 2026-06-30. SPY inception
  1993-01-29; ^GSPC pulled from 1962-01-02 for maximum cyclic-turn coverage (1962 flash break,
  1966, 1970, 1973-74, 1980-82, 1987 crash, 2000-02 dot-com, 2007-09 GFC, 2020 COVID, 2022
  bear).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [426-know-sure-thing](../426-know-sure-thing/) — Pring's own **KST**, Special K's direct
  parent: four ROCs instead of twelve, no explicit "major cyclic turn" framing (KST's pitch is
  a general trend filter). This study checks the "reduced whipsaw" upgrade claim head-to-head
  against KST on the identical tape — Special K trades 4.4× less often for a marginally better
  Sharpe, but both land significantly below buy-and-hold. **Closest cousin, tested directly.**
- [105-coppock-curve](../105-coppock-curve/) — another long-horizon momentum oscillator
  explicitly built to flag major **bear-market troughs**; its 19-signal, 76-year event-study
  design (random-timing control, not overlapping-window Welch *t*) is the template our
  Coppock-style placebo borrows. Coppock is one indicator, one direction (buy signals only);
  Special K is two ROCs' worth of machinery testing both directions.
- [425-detrended-price-oscillator](../425-detrended-price-oscillator/) — a single-scale
  detrending oscillator turned into the same long/flat timer race; strips trend out entirely
  rather than blending scales. Different construction, same "does the fancy version beat doing
  nothing" question and the same *None x Mirage* outcome.
- [427-rate-of-change](../427-rate-of-change/) — the single-scale, single-ROC building block
  Special K sums twelve copies of (at different lookbacks); its own value-add over buy-and-hold
  is also negative and insignificant (spread *t* = −1.47), a matching "beta in disguise" story
  one scale at a time.

None of the siblings sum twelve multi-scale ROCs into a crossover and test it specifically as
a **major-cyclic-turn caller** with an event study, a random-timing placebo and a matched
long/flat timer, on three independent real tapes — that combination is this study's own axis.
