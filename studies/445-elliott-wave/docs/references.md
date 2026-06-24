# References — Study 445 (Elliott Wave)

## The claim and its source

- **Elliott, R. N. (1938).** *The Wave Principle.* The original — price moves in repetitive
  five-wave impulses (1-2-3-4-5) followed by three-wave corrections (A-B-C), self-similar across
  degrees (fractal). The foundational text of the whole method.
- **Elliott, R. N. (1946).** *Nature's Law — The Secret of the Universe.* Elliott ties the wave
  counts to the Fibonacci sequence and the golden ratio.
- **Frost, A. J. & Prechter, R. R. (1978).** *Elliott Wave Principle: Key to Market Behavior.*
  The modern bible of EW; Prechter's Elliott Wave International popularised the rules we mechanise
  here — wave 2 retraces ~50–61.8% of wave 1, wave 3 is the longest/strongest leg and never the
  shortest, wave 4 does not overlap wave 1, and Fibonacci ratios govern wave extensions/targets.
- **Prechter, R. R.** *The Elliott Wave Theorist* (newsletter). The best-known practitioner forum
  for the forward-looking counts whose tradable core (the "buy wave 3" entry) this study tests.

## Why the method is irreducibly subjective (and what we tested instead)

- **The count is hindsight-dependent.** Critics (and honest practitioners) note that the "correct"
  wave count is only knowable after the move completes, and two analysts routinely label the same
  chart differently — so the theory as stated is not directly falsifiable. We therefore encode the
  **tightest mechanical version proponents accept**: a **ZigZag** swing detector for the pivots plus
  a **Fibonacci** wave-2 filter, and test only the *forward-looking* claims (wave-3 extension; a
  correction after a completed five-wave impulse).
- **ZigZag** is the standard EW labeling/charting tool (a percentage-reversal swing filter) built
  into most charting platforms; it is the least subjective way to mark the pivots a count rests on.

## Empirical / skeptical literature on Elliott Wave

- **Lo, A. W. & Hasanhodzic, J. (2010).** *The Evolution of Technical Analysis.* Context on the
  pattern-recognition tradition EW belongs to and the difficulty of out-of-sample validation.
- **Park, C.-H. & Irwin, S. H. (2007).** "What do we know about the profitability of technical
  analysis?" *Journal of Economic Surveys* 21(4). The broad survey: most rule-based TA fails to
  beat benchmarks net of costs and data-snooping corrections — the family this result joins.
- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-snooping, technical trading rule
  performance, and the bootstrap." *Journal of Finance* 54(5). Why a positive point estimate from
  one of many parameter settings (Fibonacci bands, ZigZag thresholds) is not evidence — the
  motivation for our same-bars placebo and robustness sweep.

## Shared method (this desk)

- **Newey, W. K. & West, K. D. (1987).** "A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix." *Econometrica* 55(3). The HAC *t* we report.
- **White, H. (2000).** "A reality check for data snooping." *Econometrica* 68(5). The spirit of
  the label-shuffle / same-bars-coin placebo: ask whether a random direction at the identical swing
  points would have looked as good.
- House protocol & inference bar: [`../../METHODOLOGY.md`](../../METHODOLOGY.md). The **t ≥ 2 on the
  real tape** rule for a `REAL` stamp; a synthetic control is a machinery proof, never market
  evidence.

## Related desk studies

- [`../301-triple-rsi`](../301-triple-rsi) — another viral-folklore teardown where the headline
  "win rate" turns out to be the shape of the exit, not an edge.
- [`../104-bollinger-reversion`](../104-bollinger-reversion) — a band/oscillator mean-reversion rule
  tested under the same coin-control idiom.
- [`../178-cci`](../178-cci) — the CCI oscillator's overbought/oversold rule vs a same-bars coin —
  the closest cousin in method and verdict.
- [`../440-pivot-points`](../440-pivot-points), [`../441-camarilla-pivots`](../441-camarilla-pivots),
  [`../443-volume-profile-poc`](../443-volume-profile-poc) — the same "chartist level/pattern" family
  of mechanical-proxy teardowns.
