# References & literature map — Study 939 (DRIP or Sweep)

## The claim under test

- **The DRIP orthodoxy.** Every broker's education page, every index-fund forum and
  every "power of compounding" chart says the same thing: switch automatic dividend
  reinvestment **on**, because cash sitting idle is cash not compounding. The corollary
  — that sweeping distributions to a money-market fund and reinvesting them on a
  calendar is a costly habit — is asserted far more often than it is measured. Study
  939 measures it: two investors, the same fund, the same shares on day one, differing
  only in what happens to the distribution once it lands.
- **The steelman for sweeping.** Reinvesting on a calendar batches four small tickets
  into one, keeps the tax lot count down, and — since 2022 — parks the money at 4-5%
  while it waits. If the fund's return over the delay window is no better than the bill
  rate, sweeping is free, and it is administratively tidier.
- **What is actually at stake, arithmetically.** The gap is (distribution yield) ×
  (equity-minus-cash return over the delay) × (average length of the delay). At a 2%
  yield, a 5 pp spread and a six-week average delay, that is
  0.02 × 0.05 × 0.115 ≈ **1.2 basis points a year**. The study's job is to say whether
  the tape can even see a number that small, and whether it is worth anyone's attention.

## The mechanism and its measurement

- **Cash drag.** The general result — that uninvested cash inside an otherwise fully
  invested portfolio costs the holder the risk premium on the idle balance — is the
  standard mutual-fund "cash drag" literature. See Yan (2006), *The Determinants and
  Implications of Mutual Fund Cash Holdings*, Financial Management, for the fund-level
  version; the DRIP-vs-sweep gap is the same arithmetic on a much smaller balance.
- **Dividend reinvestment and long-horizon returns.** Siegel, *Stocks for the Long Run*,
  and Dimson, Marsh & Staunton, *Triumph of the Optimists* (2002), are the canonical
  demonstrations that reinvested dividends dominate century-scale wealth. Both are
  statements about **reinvesting at all** rather than about *when* — a distinction the
  folklore routinely blurs, and the distinction this study isolates.
- **Reconstructing the distribution stream.** The identity used here,
  D_t = P_{t−1} · (TR_t / TR_{t−1}) − P_t, is the definition of an adjusted-close
  series read backwards. CRSP's `RET` / `RETX` pair is the institutional form of the
  same decomposition (total return vs return-excluding-dividends); Yahoo's
  `auto_adjust=True` / `auto_adjust=False` pair is the retail one. Our
  `dividend_reconstruction_check` scores the reconstruction against the vendor's own
  reported cash amounts as an audit.
- **Ex-date price behaviour.** Elton & Gruber (1970), *Marginal Stockholder Tax Rates
  and the Clientele Effect*, Review of Economics and Statistics — the ex-day price drop
  is close to, but not exactly, the dividend. That residual is precisely what the
  reconstruction's noise threshold has to survive, and why the reconstruction is
  audited rather than assumed.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../drip_sweep/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*,
  JASA — [`strategy.bootstrap_gap_ci`](../drip_sweep/strategy.py). The block length is
  a full quarter (63 days) because the gap is generated in bursts around pay dates; an
  i.i.d. day-level resample would badly understate its variance.

## Why the honest answer is likely to be "too small to matter"

- The quantity being raced is a *timing difference on a small cash flow*, not a return
  premium. Its natural unit is single basis points a year, which is well inside the
  band that unobservable plumbing — the pay-date lag, the broker's fractional-share
  rounding, one day of settlement — moves it by.
- The **pay lag is not on the tape.** Yahoo publishes ex-dates; the cash arrives weeks
  later on a date no free source carries. Sweeping that assumption over 0-45 days moves
  the estimate by more than the estimate itself, which is the study's central caveat.
- **Power.** The synthetic control shows that at market-realistic parameters (a 5.5%
  premium, a 3% yield, 16% vol) a single twenty-year daily tape cannot separate the
  planted effect from zero. Any confident claim in either direction, from anyone, on
  one fund's history is over-reading the data.

## Related desk studies (dedup)

- **[Study 143 — Dividend-Capture](../../143-dividend-capture/)**: trading *around* the
  ex-date to pocket the payout. That is a return-prediction claim about the ex-day price
  drop; Study 939 takes the ex-day drop as given and asks only what happens to the cash
  afterwards.
- **[Study 516 — Dividend-Month-Premium](../../516-dividend-month-premium/)**: a
  cross-sectional premium in *predicted* dividend months — a stock-selection effect, not
  an accounting policy applied to a fund you already own.
- **[Study 57 — Yield-Trap](../../57-yield-trap/)** and
  **[Study 206 — Dividend-Aristocrats](../../206-dividend-aristocrats/)**: whether
  high-dividend *stocks* beat the market. Study 939 holds the fund constant and varies
  only the reinvestment policy, so nothing here depends on which fund you own — the same
  small positive gap appears on SPY, VYM and SCHD, loosely ordered by their realised
  yields — and only loosely: raced on matched windows the ordering survives but the
  spread nearly vanishes, and per unit of yield it reverses.
- **[Study 916 — Withholding Drag](../../916-withholding-drag-international/)**: the
  other study on this desk that reconstructs the distribution stream from the two price
  legs. It asks *how much income reaches you* (a level question, international funds);
  939 asks *when you put it back to work* (a timing question, US funds), and reuses 916's
  `divs_<TICKER>_1d.parquet` cache convention.
- **[Study 934 — Lump Sum vs DCA](../../934-lump-sum-vs-dca/)** and
  **[Study 936 — Tolerance Bands](../../936-rebalance-bands/)**: the neighbouring
  cash-timing and turnover questions in this lot. Both concern *new* money or *whole
  portfolio* weights; 939 concerns only the fund's own payout stream, which is one to
  three per cent of NAV a year.

## Data sources

- **SPY, VYM, SCHD** (the payers) and **BIL** (1-3 month T-bills, the cash leg) — daily
  closes via `yfinance`, both legs: **total-return** (`auto_adjust=True`, cached as
  `prices_<TICKER>_1d.parquet`) and **price-only + distributions**
  (`auto_adjust=False, actions=True`, cached as `divs_<TICKER>_1d.parquet`). The two
  legs are cached in the shared desk cache `studies/_cache`.
- The tradable window is **SPY/VYM ∩ BIL, 2007-05-30 → 2026-06-30** and
  **SCHD ∩ BIL, 2011-10-20 → 2026-06-30**; BIL's 2007 inception gates the cash leg, so
  SPY's 1993-2007 history is excluded from the race rather than paired with a
  non-tradable cash proxy.
- **As-of 2026-06-30** — the last complete calendar month; the partial current month is
  dropped so the sample never creeps between reruns.
- **Survivorship:** all three payers are large, surviving, still-listed US ETFs chosen
  *because* they are the funds a retail holder actually owns. Nothing here is a
  cross-sectional sort, so there is no dead-fund selection in the return series — but
  the finding is a statement about these three tapes, not about a random ETF drawn in
  2007.
