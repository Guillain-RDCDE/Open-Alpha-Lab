# References & literature map — Study 934 (Lump Sum vs DCA)

## The claim under test

- **The folk advice.** "Never put a windfall in all at once — drip it in over a year.
  You buy at a lower average price, you lower your risk, and you end up ahead." It is
  the single most repeated piece of retail investment guidance, and it makes three
  separate promises: a *better average price*, *less risk*, and *more money*. Only one
  of them survives contact with a tape.
- **The steelman.** Averaging in is not a forecast; it is a rule you can keep. A
  falling market rewards it mechanically (later tranches buy cheaper), and it removes
  the single-date regret that stops people investing at all. The honest question is not
  whether it *ever* wins but what it costs on average, and whether there is a
  *conditional* state — a stretched market, or the middle of a drawdown — where the
  advice comes good.
- **What makes this test different from the popular ones.** Almost every published
  version leaves the uninvested balance earning **nothing**. That is an assumption, not
  a tape: over 2007-2026 the T-bill leg paid 0% for six years and roughly 5% for three.
  Here the waiting money sits in **BIL** and earns its actual total return, so the cash
  drag being measured is the *real* one.

## The prior literature

- **Constantinides (1979), *A Note on the Suboptimality of Dollar-Cost Averaging as an
  Investment Policy*, Journal of Financial and Quantitative Analysis 14(2).** The
  theoretical result: DCA is dominated as a policy under standard preferences, because
  it commits to a deterministic path that ignores information. Not an empirical claim —
  which is why it needs a tape test.
- **Vanguard (2012, updated 2023), *Cost Averaging: Invest Now or Temporarily Hold Your
  Cash?*** The canonical empirical study: across US, UK and Australian tapes and a
  rolling-window design close to ours, immediate investment finishes ahead of a
  12-month averaging-in schedule roughly **two thirds** of the time. Our SPY window is
  more lopsided than that; our long-history SPY window is close to it.
- **Rozeff (1994), *Lump-Sum Investing versus Dollar-Averaging*, Journal of Portfolio
  Management 20(2).** Frames the result as mechanical: with a positive expected return,
  time out of the market has a price, and DCA buys its lower volatility by holding a
  lower average equity weight — an *exposure* choice dressed as a *timing* choice.
  **This is the paper the study's second stamp rests on**: Rozeff's claim is testable, so
  we test it directly with the exposure-matched control (race DCA against a *static*
  portfolio holding its own analytic average weight, (n+1)/2n). On this tape Rozeff is
  right to the cent — the matched gap is −0.04c with *t* = −0.08 — which is why the
  Tradability stamp is Mirage rather than Investable: there is no timing edge left to
  bank once the beta is equalised.
- **Williams & Bacon (1993), *Lump Sum Beats Dollar-Cost Averaging*, Journal of
  Financial Planning.** An early US-only version of the same finding.
- **Statman (1995), *A Behavioral Framework for Dollar-Cost Averaging*, Journal of
  Portfolio Management.** The defence that actually holds: DCA is prospect-theoretic
  regret control, not an optimiser's answer. It is bought with expected wealth, and it
  is sometimes worth the price. This study prices it; it does not tell anyone whether
  the price is worth paying.
- **Shtekhman, Tasopoulos & Wimmer (2012)** and the practitioner literature on
  *sequence-of-returns risk* explain why the dispersion cut is real even where the mean
  gap is not: DCA truncates the left tail of *entry-date luck* at the cost of the mean.

## Related desk studies (dedup)

- **[Study 101 — Slow-and-Steady](../../101-slow-and-steady/)**: the desk's first pass at
  this question. It rolls the race over **every trading day**, with idle cash earning a
  flat **0%**, on SPY alone, and stamps the *DCA-beats-lump-sum* claim None/Mirage.
  Study 934 re-asks it from the other side and with the tape the first pass assumed
  away: idle cash earns **BIL's real total return**, starts are **month-ends** (the
  schedule a real drip actually follows), and the new content is the **exposure-matched
  control** (is the gap timing or beta?) and the **conditional** question — does DCA come
  good when you start from a stretched market or from inside a drawdown? — plus a
  **bond-heavy** variant, a tranche-length sweep and a fixed-ticket cost sweep. Same
  family, different question, and the two Tradability stamps land in the same place
  (**Mirage** both times): 101 says DCA's supposed superiority is a mirage, 934 says the
  lump sum's counter-advantage is one too. Neither schedule has an edge; the difference
  between them is exposure.
- **[Study 241 — Buy-the-Dip](../../241-buy-the-dip/)**: holding cash until a drawdown
  *threshold* fires. That is a conditional entry rule; DCA is an unconditional schedule
  that never looks at the price.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)** and
  **[Study 97 — Balancing-Act](../../97-balancing-act/)**: what to do with a portfolio
  you already hold. This study is about the one-off decision of how to *get in*.
- **[Study 836 — Rebalance Timing Luck](../../836-timing-luck/)**: how much of a
  strategy's Sharpe is the arbitrary choice of rebalance date. The dispersion half of
  this study is the same phenomenon seen from the investor's side — entry-date luck.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../lump_vs_dca/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). Monthly starts
  with twelve-month horizons overlap by up to eleven months, so 12 lags is the natural
  correction; the non-overlapping check (every twelfth start, averaged over all twelve
  phases) is the belt to that braces.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*,
  JASA — [`strategy.block_bootstrap_mean_ci`](../lump_vs_dca/strategy.py), 12-month
  blocks so a whole horizon of overlap is resampled intact.
- **Wilson score interval.** Wilson (1927), *Probable Inference, the Law of Succession,
  and Statistical Inference*, JASA — [`strategy.wilson_interval`](../lump_vs_dca/strategy.py),
  used for the win-rate interval (the normal approximation misbehaves near the tails).

## Data sources

- **SPY** (US large-cap equity), **IEF** (7-10y US Treasuries, the bond-heavy variant)
  and **BIL** (1-3 month T-bills, the cash leg) — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`), read from the shared desk cache. The headline window
  is the SPY∩BIL common sample, which starts at BIL's 2007 inception: before that there
  is no live T-bill ETF to credit the waiting cash with, so the long-history extension is
  run under the explicit **0% cash ASSUMPTION** and labelled as such. That extension is
  floored at a **pinned 2000-01-03** rather than at SPY's inception, because
  `studies/_cache` is shared and other studies re-pull SPY with their own start dates —
  the depth of the cached tape is not this study's to assume. Fingerprints in
  `docs/results.md` identify the pull that produced the run and move with each refresh;
  the headline reads did not.
- **Non-tape inputs, all labelled.** The valuation "stretch" tercile is a **price-based
  proxy** (level vs trailing three-year mean), not CAPE — no earnings data is used in
  this study. The fixed-ticket sweep assumes a **$10,000 windfall** and a **$0-$10**
  commission bracket. Both are swept rather than asserted.
- **As-of 2026-06-30.** The partial current month is dropped, and only windows whose
  twelfth month-end falls on or before the as-of are kept, so the sample never creeps
  and no window is scored on a partial year.
