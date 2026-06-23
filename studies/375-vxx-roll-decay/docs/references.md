# References & literature map — Study 375 (VXX-Roll-Decay)

## The claim under test

- **The pitch.** Shorting VIX-futures ETPs (`VXX`, `VIXY`, `UVXY`) is the canonical retail
  "short-vol carry": because the VIX term structure is in **contango** the large majority of
  the time, a constant-maturity short-dated VIX-futures product must continually **roll** out
  of a cheaper near future into a richer far one, bleeding value — so a short harvests a
  steady **roll yield**. The decay is dramatic: VIXY's split-adjusted price fell by ~5 orders
  of magnitude over 2011-2026. The counter-claim — *"picking up nickels in front of a
  steamroller"* — is equally folklore: the short is implicitly **selling crash insurance**
  and gets run over when vol spikes.
- **The roll-decay mechanism.** Documented across the volatility-ETP literature: the products
  track an index (e.g. the *S&P 500 VIX Short-Term Futures Index*) that holds a weighted
  blend of the front two VIX futures rolled daily to a constant ~30-day maturity. See the
  ProShares/Barclays prospectuses and the **CBOE VIX** white paper (Cboe, *The CBOE
  Volatility Index — VIX*, 2009/2019) for the construction.

## The economics — contango, roll yield, and the variance risk premium

- **The variance risk premium (the deep source of the carry).** Carr & Wu (2009),
  *Variance Risk Premiums*, Review of Financial Studies — implied variance trades rich to
  realized variance on average, so sellers of variance/vol earn a premium for bearing
  crash risk. Bollerslev, Tauchen & Zhou (2009), *Expected Stock Returns and Variance Risk
  Premia*, RFS. The VXX short is a (noisy, lossy) way to harvest this premium.
- **VIX-ETP roll cost specifically.** Whaley (2013), *Trading Volatility: At What Cost?*,
  Journal of Portfolio Management — quantifies the structural drag in VXX from the roll and
  contango. Alexander & Korovilas (2012), *Diversification of Equity with VIX Futures*, and
  Eraker & Wu (2017), *Explaining the Negative Returns to Volatility Claims*, Journal of
  Financial Economics — why long VIX-ETPs have deeply negative expected returns (hence the
  short's positive carry).
- **The crash tail / "steamroller."** Short-vol books carry extreme **negative skew** and
  **fat tails**: the Feb-5-2018 **"Volmageddon"** wiped out the XIV ETN (−96% in a day) and
  forced its termination; COVID (Mar-2020) and the Aug-5-2024 yen-carry unwind repeated the
  pattern. The classic framing of selling-insurance return profiles: Taleb (2004),
  *Fooled by Randomness*; and the option-writing analogue in Bondarenko (2014), *Why Are Put
  Options So Expensive?*

## Why VIXY (not VXX), and how we measure

- **Vehicle choice.** A continuous, split-adjusted yfinance `VXX` series only starts in 2018
  (the iPath ETN matured and was reissued Jan-2018), which would *exclude Volmageddon*. We
  therefore short **VIXY** (ProShares VIX Short-Term Futures ETF), which tracks the **same**
  short-dated constant-maturity VIX-futures index, has an unbroken tape from 2011, and
  **contains** all three tail events. This is named on the Signal axis as a VXX-equivalent,
  not a fabrication: every input is a public adjusted close.
- **Contango gauge.** We proxy the term-structure slope with `^VIX3M / ^VIX` (3-month vs
  spot VIX); > 1 ⇒ contango ⇒ roll tailwind. (A direct futures-curve slope would be cleaner;
  the 3M/spot ratio is the free, transparent stand-in.)

## Inference & method lineage (the desk's shared engine)

- **HAC (Newey-West) inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica — carry
  returns cluster in vol regimes, so the Signal-axis *t* uses a Bartlett-kernel HAC SE
  ([`strategy.hac_t`](../vxx_roll_decay/strategy.py)); an i.i.d. *t* would overstate
  significance.
- **Sign-shuffle placebo null.** A distribution-free randomization test
  ([`strategy.placebo_pvalue`](../vxx_roll_decay/strategy.py)) — Fisher's randomization
  logic; Efron & Tibshirani (1993), *An Introduction to the Bootstrap*.
- **Tail diagnostics.** Skew, excess kurtosis, single-worst-day, 1% VaR/CVaR and the
  compounded drawdown ([`strategy.perf`](../vxx_roll_decay/strategy.py)) — the objects that
  decide tradability for any short-insurance book (Artzner et al., 1999, *Coherent Measures
  of Risk*, on CVaR).
- **Costs with borrow.** One-way rebalancing turnover **and** a short-borrow financing fee
  ([`strategy.net_of_costs`](../vxx_roll_decay/strategy.py)) — shorts pay borrow, and VIX
  ETPs become hard-to-borrow in exactly the crises when the carry would pay.
- **Deterministic synthetic control.**
  [`data.synthetic_etp`](../vxx_roll_decay/data.py) plants a known daily carry knob and
  injects crashes; with the carry set to zero the engine must NOT find a positive carry, and
  the injected crash always reproduces the negative skew — a machinery proof, never market
  evidence.

## Data sources used here

- **yfinance** daily adjusted closes for VIXY, `^VIX`, `^VIX3M`, `^VIX9D`, SPY,
  2011-01-04 → 2026-06-22, cached under `_cache/vol_prices.csv`. All headline numbers are
  pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- The **short-vol / variance-premium family** more broadly: any study that sells convexity
  for carry shares this skew/kurtosis pathology — a real premium that a single tail event can
  erase. The lesson here (real carry, fragile book, no free lunch) is the template for "I
  earn a steady yield" trades whose risk lives entirely in the left tail.
