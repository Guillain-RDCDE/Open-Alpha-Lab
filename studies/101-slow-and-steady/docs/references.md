# References & literature map — Study 101 (Slow-and-Steady)

## The claim under test

Dollar-cost averaging (DCA) is the single most-repeated piece of personal-finance advice:
*"don't invest your windfall all at once — spread it in over a year. You'll buy at a lower
average price, you'll lower your risk, and you'll beat going all-in."* The strong,
sold-at-full-strength version bundles two promises into one: DCA is both **safer** (lower
risk) **and smarter** (higher terminal wealth) than a lump sum.

- Popular framing, e.g. Investopedia, *"Dollar-Cost Averaging (DCA)"*:
  <https://www.investopedia.com/terms/d/dollarcostaveraging.asp>
- It is the default advice of most robo-advisors, employer 401(k) plans (by construction),
  and personal-finance media whenever someone receives a bonus, inheritance, or rollover.

## Why the steelman is almost coherent

- **The arithmetic of averaging is real.** When you buy a *fixed dollar amount* each period,
  you mechanically buy more shares when prices are low and fewer when high, so your average
  cost per share is the *harmonic* mean of prices — always ≤ the arithmetic mean. On a
  **falling or sideways** tape this genuinely lowers your cost basis and DCA out-earns the
  lump sum. The folklore is built on this true special case.
- **DCA lowers the dispersion of outcomes.** Spreading entry across many dates diversifies
  away the single-date entry-timing risk, so the spread of terminal wealth is narrower —
  a genuine, measurable risk reduction (and a behavioural hedge against the regret of going
  all-in the day before a crash).

## Why it fails *as stated* ("beats a lump sum")

- **Markets rise on average, so waiting costs money.** Vanguard, *Cost averaging: invest now
  or temporarily hold your cash?* (2012; reissued 2023) studies rolling windows across the
  US, UK and Australian markets and finds **lump-sum beats DCA roughly two-thirds of the
  time** — because the expected return of being invested is positive, every dollar left in
  cash while it "averages in" is a dollar forgoing the equity risk premium.
  <https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf>
- **DCA is provably sub-optimal for an expected-utility maximiser.** Constantinides (1979),
  *A Note on the Suboptimality of Dollar-Cost Averaging as an Investment Policy*, Journal of
  Financial and Quantitative Analysis 14(2), 443–450, shows DCA is dominated as a *sequential*
  policy: it ignores information and deliberately holds a lower-than-optimal equity weight
  early. The "lower average cost" intuition does not survive once you account for the
  opportunity cost of the un-deployed cash.
- **The risk reduction is just lower average exposure.** A DCA path holds, on average, only
  about half its capital in the market during the deployment year. A like-for-like risk
  comparison must therefore separate "lower risk because better-timed" (false) from "lower
  risk because less invested" (true) — the same alpha-vs-beta confusion the desk meets again
  and again.

## Method lineage

- **Newey–West HAC standard errors** for the mean of a *heavily overlapping* (and therefore
  strongly autocorrelated) series of rolling-window gaps: Newey & West (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, Econometrica 55(3), 703–708. A naive t-stat on overlapping windows is badly
  overstated; HAC with a long lag is the honest read.
- **Wilson score interval** for the lump-sum win-rate proportion: Wilson (1927), *Probable
  Inference, the Law of Succession, and Statistical Inference*, JASA 22, 209–212.
- **Synthetic drift control** — a deterministic geometric random walk with a single drift
  knob is the study's two-sided positive control: negative drift plants a DCA win, positive
  drift plants a lump-sum win, and the harness must bank whichever is true.

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. Total return is the only fair tape
  here: the dollars must compound identically whether deployed all at once or fed in monthly.
  Cash awaiting deployment is assumed to earn **0%** — a stated, conservative-to-DCA choice
  (a T-bill would lift DCA only marginally and would not flip the verdict).

## Related desk studies

- [Study 91 — Death-Cross](../../91-death-cross/) — the canonical "real risk reduction, but
  it's lower beta, not skill" split this study mirrors.
- [Study 68 — All-Weather](../../68-all-weather/) — risk reduction that *is* the product, and
  the bar a de-risking story must clear.
