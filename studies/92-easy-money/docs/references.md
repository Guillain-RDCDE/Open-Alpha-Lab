# References & literature map — Study 92 (Easy-Money)

## The claim under test

VIX futures spend most of their life in **contango** — the further-dated future trades
above the nearer one, which trades above spot VIX. A constant-maturity long-volatility
ETP (VXX, VIXY, …) holds the front two futures and **rolls** daily from the cheaper
expiring contract into the pricier next one, *selling low and buying high* by
construction. That roll is a steady drag — the ETP "bleeds" lower nearly every day. The
strong, sold-at-full-strength version of the folklore is: *"just short the long-vol ETP
(or hold an inverse like SVXY) and pocket the roll — it's near-free money, an almost
risk-less carry."*

- Popular framing, e.g. Investopedia, *"Contango"* and *"VIX"* explainers, and the
  perennial "short VXX" trade in retail forums:
  <https://www.investopedia.com/terms/c/contango.asp>
- The inverse ETPs themselves (SVXY, and the defunct **XIV**) were marketed as a way to
  monetise exactly this decay.

## Why the steelman is almost coherent

- **The roll-yield / term-structure premium is real and documented.** The VIX-futures
  curve is upward-sloping the large majority of the time, and a short-front-month
  position has historically earned a positive average carry (Alexander & Korovilas,
  *Understanding ETNs on VIX Futures*, 2012; Whaley, *Trading Volatility: At What Cost?*,
  Journal of Portfolio Management 2013, which documents VXX's structural decay).
- **Selling variance/volatility earns a premium across markets** — the variance risk
  premium is one of the most robust facts in empirical finance (Carr & Wu, *Variance Risk
  Premiums*, RFS 2009; Bollerslev, Tauchen & Zhou, *Expected Stock Returns and Variance
  Risk Premia*, RFS 2009). Shorting a long-vol ETP is one way to harvest it.
- So the *carry is genuine* — the decay of VXX/VIXY is one of the most reliable
  large negative drifts in any liquid security.

## Why it is NOT free money ("near-risk-less carry" is the part that fails)

- **You are short crash insurance, not collecting a free coupon.** The premium is the
  price others pay to be *long* volatility as a hedge; you earn it by agreeing to pay out
  exactly when volatility explodes. The return is steady-drip-then-catastrophe — a
  classic short-left-tail, negatively-skewed, fat-kurtosis payoff.
- **Volmageddon, 5 February 2018.** The inverse ETN **XIV** (Credit Suisse) lost ~96% of
  its value in a single afternoon as VIX nearly doubled (VIX 17→37), triggering its
  acceleration/termination clause; it was wound down. SVXY survived only by halving its
  leverage afterward. (Contemporaneous coverage: the SEC/press post-mortems on the
  February 2018 "volmageddon".)
- **March 2020 (COVID)** drove spot VIX to its all-time closing high of **82.69**
  (2020-03-16); **August 2024** (the yen carry-unwind) spiked VIX intraday above 60. Each
  episode handed a short-vol book a one-day loss that can erase years of accumulated carry.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return series:
  Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica. We require a HAC *t* over
  **2** on the real tape before stamping the carry's Signal `REAL`.
- **Tail diagnostics** — skewness, excess kurtosis, worst single-day return and maximum
  drawdown are reported alongside the Sharpe, because the Sharpe of a short-insurance
  payoff systematically *flatters* it (a steady mean and a hidden left tail).
- **Survivable-sizing** — leverage is solved so the realised worst up-day of the ETP
  costs at most a fixed fraction of capital, isolating what the carry is worth once the
  position is small enough not to be wiped out by the spike.

## Data sources used

- **VIXY** (ProShares VIX Short-Term Futures ETF, the long-vol ETP), daily,
  **total-return adjusted** via `quantlab.data` (Yahoo Finance), cached to parquet under
  `_cache/`. VIXY has undergone several **1:4 reverse splits** (its raw price would
  otherwise jump at each); total-return adjustment folds those back in so the series is a
  single continuous NAV-equivalent — stated as a decision, not a detail. Shorts pay
  borrow, charged daily.
- **^VIX** (CBOE Volatility Index spot), daily, since 1990 — used for the term-structure
  context and to date the spikes (2018, 2020, 2024). An index is not tradable and carries
  no dividends/splits, so its adjustment mode is a no-op.
- Window is **VIXY-bounded: 2011-01-04 → 2026-06-12** (the ETP's full life). Stated
  openly; the carry conclusion is about the post-2011 ETP era.

## Related desk studies

- [Study 86 — Tail-Radar](../../86-tail-radar/) — the CBOE SKEW index and whether the
  tail can be *seen* coming (here we are *short* that tail).
- [Study 69 — Safe-Haven](../../69-safe-haven/) — the other side of the crash-hedge trade.
- [Study 72 — Loaded-Dice](../../72-loaded-dice/) — the "backtests great until the regime
  turns" pattern.
