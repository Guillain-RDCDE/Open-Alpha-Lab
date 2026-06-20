# References & literature map — Study 316 (Bank-Failure)

## The claim under test

- **"Banking blow-ups are a buy signal" vs "the first domino."** The folk wisdom splits
  two ways. The contrarian camp — popularised by Baron Rothschild's apocryphal *"buy when
  there's blood in the streets"* and by every "be greedy when others are fearful" (Warren
  Buffett, 2008 NYT op-ed *Buy American. I Am.*) — says a headline bank failure is peak
  panic, hence a buy. The cautious camp says a bank failure is a *systemic warning* — the
  visible first domino of contagion (Lehman → the GFC). We take both literally and run an
  **event study** of SPY and the financials sector (XLF) around a hardcoded table of
  public-knowledge failure dates: Continental Illinois (1984), Northern Rock (2007), Bear
  Stearns / IndyMac / Lehman / WaMu / Wachovia (2008), MF Global (2011), and the 2023
  cluster (SVB, Signature, Credit Suisse, First Republic).

## Event-study methodology (the design)

- **Brown & Warner (1985)**, *Using daily stock returns: The case of event studies*
  (Journal of Financial Economics). The canonical reference for daily-return event studies:
  abnormal returns, cumulative abnormal returns (CARs), and the small-sample/clustering
  pitfalls that dominate this study (with 11–12 clustered events, the standard tests are
  weak by construction).
- **MacKinlay (1997)**, *Event Studies in Economics and Finance* (Journal of Economic
  Literature). The textbook treatment of the CAR estimator and its inference — and the
  warning that **event clustering** (overlapping windows, common calendar) breaks the
  cross-sectional independence the standard *t*-test assumes. The 2008 cluster here is the
  textbook example.
- **Boehmer, Musumeci & Poulsen (1991)**, *Event-study methodology under conditions of
  event-induced variance* (JFE). Why event-day volatility inflation makes naive tests
  over-reject — relevant to why we lean on a placebo/permutation control and a HAC *t*
  rather than a raw cross-sectional *t*.

## The crisis-contagion literature (the "first domino" camp)

- **Reinhart & Rogoff (2009)**, *This Time Is Different: Eight Centuries of Financial
  Folly* (Princeton UP). Banking crises are typically protracted and contagious — the
  intellectual case that a bank failure can be a leading warning, not a bottom.
- **Gorton (2010)**, *Slapped by the Invisible Hand: The Panic of 2007*. The mechanics of
  the 2008 run on repo — why the autumn-2008 failures clustered and cascaded, which is
  precisely the cluster that drives this study's bearish average.
- **Bernanke (1983)**, *Nonmonetary Effects of the Financial Crisis in the Propagation of
  the Great Depression* (American Economic Review). The original macro case for bank
  failures amplifying real downturns.

## The contrarian / mean-reversion camp (the "buy the blood" thesis)

- **De Bondt & Thaler (1985)**, *Does the Stock Market Overreact?* (Journal of Finance).
  The overreaction hypothesis — extreme bad news overshoots and reverses. A bank-failure
  panic is the steelman case for an overreaction bounce.
- **The 2023 episode.** The SVB / Signature / Credit Suisse / First Republic cluster — a
  167-year-old G-SIB (Credit Suisse) forced into a UBS takeover in ten days — produced a
  *positive* +20-day SPY move, the cleanest modern test of the "buy the blood" view, and
  the data point that most separates 2023 from 2008 in our results.

## The headline trap — win-rate vs expectancy, and the cluster confound

- **Win-rate vs expectancy.** Our overlay wins 67% of the time yet *loses* money on
  average — many small rebounds, a few catastrophic 2008 trades. This is the same
  exit-asymmetry illusion dissected on this desk in
  [Study 301 — Triple-RSI](../../301-triple-rsi/) and
  [Study 72 — Loaded-Dice](../../72-loaded-dice/): a high hit-rate is not an edge.
- **The clustering confound.** A naive permutation test against random dates returns a
  "significant" bearish result — but it is really detecting *"were these dates in 2008?"*,
  not *"is a bank failure a signal?"* Per the desk's inference bar, a synthetic/placebo
  control is a **machinery proof, never market evidence**; the per-event HAC *t* (−0.93)
  decides, and it is below 2.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../bank_failure/strategy.py).
- **Block bootstrap.** Künsch (1989) / Politis & Romano (1992) — circular block bootstrap
  for the CAR CI, which preserves the temporal clustering of events
  ([`strategy.block_bootstrap_ci`](../bank_failure/strategy.py)).
- **Permutation / placebo control.** The random-date placebo arm
  ([`data.random_dates`](../bank_failure/data.py),
  [`strategy.permutation_pvalue`](../bank_failure/strategy.py)) and a deterministic
  positive-control tape with a planted post-event drift
  ([`data.synthetic_tape`](../bank_failure/data.py)).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), `auto_adjust=True` (total-return-ish):
  SPY back to 1993, XLF back to 1998. Event dates are public-knowledge announcement /
  seizure / forced-sale days (calendar-known → no extra execution lag). All headline
  numbers are pinned with an as-of date and content fingerprint
  ([`docs/results.md`](results.md)). The offline reproducible core and the test-suite run
  on the deterministic [`data.synthetic_tape`](../bank_failure/data.py) generator, never
  the network.

## Related desk studies

- **[Study 241 — Buy-the-Dip](../../241-buy-the-dip/)**: the generic "buy the dip"
  question (None / Mirage). Bank-Failure is the *event-conditioned* cousin — buy a
  specific kind of dip (a banking blow-up) — and reaches the same place for the same
  reason (no certifiable edge, dominated by a few episodes).
- **[Study 167 — Hindenburg-Omen](../../167-hindenburg-omen/)** and
  **[Study 160 — Skyscraper-Curse](../../160-skyscraper-curse/)**: other "this signals a
  crash" omens, both None / Mirage. Same small-sample, hindsight-clustered failure mode.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)** / **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**:
  where the win-rate-vs-expectancy illusion was dissected on this desk.
