# References & literature map — Study 837 (Look-Ahead Standardization)

## The claim, at full strength

The pitfall: *"z-score your features before modelling."* Everyone does it —
`x = (x - x.mean()) / x.std()` — and almost everyone, at least once, runs it **over the entire
sample**, before splitting into train and test. That single line computes the mean and standard
deviation from data that includes the **future**, so every historical feature value is standardised
with information the researcher could not have had at the time. It is the most common accidental
**look-ahead bias** in a quantitative pipeline, and because it hides inside "preprocessing" it slips
past the audit that a lag-shifted signal would never survive. This study makes the trap undeniable by
running it on synthetic worlds we *built* to carry **no point-in-time-tradeable edge**, so any IC or
Sharpe the full-sample z-score prints is, by construction, a leak.

## Look-ahead bias and data snooping — the source literature

- **López de Prado (2018)**, *Advances in Financial Machine Learning*, Wiley. The modern reference on
  leakage in financial ML: features and labels must be constructed from **point-in-time** information,
  and any transform (scaling, imputation, feature selection) fit on the whole sample **contaminates
  the backtest**. Chapters on the dangers of the naive train/test split and on why cross-validation
  leaks when preprocessing precedes the fold split are the direct charter for this study.
- **Kaufman, Rosset & Perlich (2012)**, *"Leakage in Data Mining: Formulation, Detection, and
  Avoidance."* *ACM Transactions on Knowledge Discovery from Data* 6(4). The canonical taxonomy of
  data leakage; **normalisation/standardisation fit on train+test together** is a textbook example of
  their "no time-machine" rule being violated. Not finance-specific, which is the point — the pitfall
  is universal.
- **Bailey, Borwein, López de Prado & Zhu (2014)**, *"Pseudo-Mathematics and Financial Charlatanism:
  The Effects of Backtest Overfitting on Out-of-Sample Performance."* *Notices of the AMS* 61(5). Why
  an in-sample statistic (here, the leaked IC/Sharpe) says nothing about out-of-sample performance
  when the sample was used to build the signal — the inference bar this desk enforces.
- **Arnott, Harvey & Markowitz (2019)**, *"A Backtesting Protocol in the Era of Machine Learning."*
  *Journal of Financial Data Science* 1(1). A practitioner protocol that explicitly warns against
  full-sample scaling/normalisation and prescribes expanding-window, point-in-time feature
  construction — the "honest" method this study validates.

## Non-stationarity — why the leak bites here and not everywhere

- **Random walks and unit roots.** The leak is largest precisely when the feature is
  **non-stationary** (a random walk / integrated series), because its full-sample mean is a
  future-dependent quantity the expanding mean never converges to. For a stationary feature,
  full-sample standardisation is nearly a per-name affine rescale and leaks little — the contrast this
  study draws. See any time-series text (e.g. **Hamilton, 1994**, *Time Series Analysis*, Princeton)
  on the divergence of the sample mean of an I(1) process.
- **Information Coefficient (IC).** **Grinold & Kahn (2000)**, *Active Portfolio Management*, 2nd ed.,
  McGraw-Hill. The rank IC (cross-sectional correlation of signal and forward return) is the standard
  yardstick for a predictive feature; here it is the number the leak inflates.

## Method — the honest yardstick and the machinery proof

- **Expanding / rolling standardisation.** Compute every normalisation statistic on a **past-only**
  window (the one you could run live). This study's `expanding_standardize` reads ~0 on both nulls and
  recovers a planted edge — an unbiased detector.
- **Newey & West (1987)**, *"A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix."* *Econometrica* 55(3). The HAC *t* on the daily IC
  series, robust to the strong autocorrelation the random-walk feature induces.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — a synthetic control is a *machinery
  proof*, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real tape, which a synthetic-only
  demo can never provide; costs are charged one-way × NAV; any synthetic-dependent claim averages over
  ≥ 20 seeds.

## Neighbours on this bench (the dedup map)

- **[Study 347 — Look-Ahead Bias](../../347-look-ahead-bias/)** — the **generic** look-ahead:
  aligning a signal to a return it could not have observed (a mis-timed shift). Study 837 is the
  **specific normalisation-leakage** case: the signal is correctly *timed*, but the **standardisation
  statistic** used to build it is fit on the full sample, leaking the future through the preprocessing
  rather than the alignment.
- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)** — inflating a Sharpe by
  **searching** many rules and reporting the luckiest (a multiple-comparisons artefact). 837 inflates
  a Sharpe with **no search at all** — a single mis-specified preprocessing step on one feature.
- **[Study 590 — Sharpe-Hacking](../../590-sharpe-hacking/)** — inflating a *reported* Sharpe by
  **transforming the returns** (smoothing/leverage/vol-target). 837 inflates the *predictive* IC/Sharpe
  by transforming the **feature** with future-inclusive statistics — the same family of "the number is
  an artefact of how it was computed," corrected by point-in-time construction rather than an
  autocorrelation adjustment or a trial count.
- **[Study 831 — Gold Real-Yield Timing](../../831-gold-real-yield-timing/)** — a nearby
  timing/backtest study; a useful reminder that even a *correctly* preprocessed signal has to clear
  the inference bar. 837's failure is upstream of all of that — in the feature-scaling step itself.
