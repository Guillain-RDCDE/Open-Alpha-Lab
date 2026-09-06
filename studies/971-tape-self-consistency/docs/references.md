# Sources & literature map — Study 971 (Does the Tape Agree With Itself?)

## Data errors are a research risk, not an IT problem

- **Ince, O. S. & Porter, R. B. (2006), "Individual Equity Return Data from Thomson Datastream:
  Handle with Care!", *Journal of Financial Research* 29(4), 463-479.** The canonical warning:
  uncleaned vendor data produces spurious return anomalies, particularly in small caps. Their
  screening rules are the ancestor of every audit like this one.
- **Rosenberg, J. V. & Houglet, M. (1974), "Error Rates in CRSP and Compustat Data Bases and
  Their Implications", *Journal of Finance* 29(4), 1303-1310.** Fifty years old and still the
  clearest statement that the reference databases themselves disagree.
- **Canina, L., Michaely, R., Thaler, R. & Womack, K. (1998), "Caveat Compounding: A Warning
  About Using the Daily CRSP Equal-Weighted Index to Compute Long-Run Excess Returns",
  *Journal of Finance* 53(1), 403-416.** Compounding a daily series into a long-horizon one is
  exactly where feeds and researchers diverge — the subject of check 1.
- **Shumway, T. (1997), "The Delisting Bias in CRSP Data", *Journal of Finance* 52(1),
  327-340.** What is *absent* from a feed matters more than what is wrong in it.

## Corporate actions and adjusted prices

- **CRSP, *Data Definitions and Calculations* (current edition).** The reference definition of a
  total-return series: reinvest distributions at the close of the ex-date, apply split factors
  to shares. Check 2 implements exactly this and compares.
- **Bali, T. G., Engle, R. F. & Murray, S. (2016), *Empirical Asset Pricing: The Cross Section
  of Stock Returns*, ch. 2-3.** Practical treatment of adjustment factors and their failure
  modes.
- **Yahoo! Finance / `yfinance` documentation and issue tracker.** The provider used here
  publishes `auto_adjust`, `Adj Close` and an `actions` event stream; the relationship between
  them is documented informally, which is a reason to test rather than to assume.

## Exchange calendars

- **NYSE, "Holidays & Trading Hours"**, and the `pandas_market_calendars` project. The
  reference used in check 3 is inferred from the sample rather than imported, so that the study
  has no dependency the reader must install — with the stated cost that a session missing from
  *every* tape is invisible.

## Neighbours on this desk

**347-look-ahead-bias**, **345-survivorship-bias**, **917-nav-staleness-timezone**,
**972-adjustment-mode-matters**, **919-index-methodology-change**, **963-half-day-sessions**
(which uses the volume tape to *confirm* a calendar rather than to audit it).
