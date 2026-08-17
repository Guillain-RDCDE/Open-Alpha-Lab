# References & literature map — Study 956 (the ADR custody fee)

## The claim under test

- **The pass-through depositary fee.** An American Depositary Receipt is issued by a
  depositary bank (BNY Mellon, Citi, Deutsche Bank, JPMorgan) that holds the underlying
  foreign shares. Since the SEC permitted it in 2009, the depositary charges holders an
  annual **custody / depositary service fee**, published in the deposit agreement and
  typically **$0.01–$0.05 per ADS per year**, usually netted out of a dividend rather than
  billed. The claim under test is the folk version: *this invisible fee measurably erodes an
  ADR's total return relative to owning the same company at home*.
- **The steelman.** Both legs are claims on identical cash flows, so in dollars their
  total-return indices can only diverge by the leaks the wrapper imposes: the custody fee,
  the foreign dividend withholding tax, and the depositary's FX conversion spread. Each is
  a *drift*, not a wiggle, so a twenty-year trend fit should see it even though a single
  day's noise is a hundred times larger. That is exactly the estimator this study builds.
- **Primary sources for the schedules.** The deposit agreements and fee disclosures filed
  on SEC Form F-6 (and the depositaries' own fee tables — e.g. BNY Mellon's ADR fee
  disclosure, Citi's ADR fee schedules) are where the 1–5 cent numbers come from. They are
  the study's **benchmark**, never an input to it: the tape is asked what it can see on its
  own, and only afterwards compared with the published rate.

## Why the effect should exist — the mechanism

- **Depositary economics.** Kim, Szakmary & Mathur (2000), *Price transmission dynamics
  between ADRs and their underlying foreign securities*, Journal of Banking & Finance —
  ADR and home-line prices are cointegrated with the exchange rate, i.e. the price ratio is
  a *stationary band*, which is precisely why a slow drift in the ratio is identifiable and
  a drift in daily returns is not.
- **Gagnon & Karolyi (2010), *Multi-market trading and arbitrage*, Journal of Financial
  Economics** — cross-listed price deviations are small (median well under 1 %), mean
  reverting, and bounded by conversion frictions. Their result is the reason the *price*
  ratio works as a placebo here: whatever the ADR wrapper costs, it is not showing up in
  the share price.
- **Karolyi (1998, 2006), *Why do companies list shares abroad?* / *The world of
  cross-listings*, Review of Finance** — the survey framing of the ADR wrapper as a bundle
  of convenience services with an explicit price.
- **Foreign dividend withholding.** Desai & Dharmapala (2011) and the OECD model-treaty
  literature on relief-at-source versus reclaim explain why the *rate an ADR holder actually
  suffers* is not a constant and not observable from prices — the reason this study labels
  the withholding rate an assumption and sweeps it rather than reporting a split.

## Why it can fail to be measurable

- **The signal-to-noise problem.** Non-synchronous closes (the home market shuts hours
  before New York) put ~1–2 % of daily noise around a fee worth 0.1 % *a year*. Any estimator
  that averages daily return differences is hopeless here; the study uses the *level* trend
  instead, which is the standard fix for a stationary-error regression (Newey & West's HAC
  covariance with a long lag, because the error is close to a unit-root process).
- **Corporate actions.** ADS ratio changes, rights issues and spin-offs put permanent steps
  in the ratio that dwarf the fee. Bailey, Karolyi & Salva (2006), *The economic consequences
  of increased disclosure*, Journal of Financial Economics, documents how disruptive these
  events are to cross-listed comparisons. Our level-shift detector exists for them.
- **The fee may not be collected where the tape can see it.** Depositaries collect the
  service fee two ways: netted out of a dividend, or charged straight to the holder's
  account through DTC as a separate line item (which is also how holders of non-dividend
  paying ADRs are billed). Only the first route touches a total-return series. If the
  vendor records the *declared gross* per-ADS rate — which is what this study's withholding
  sweep proves it does — then even the netted route may be invisible. This is why the
  measured gap is reported as an **upper bound on** the fee and never as the fee itself,
  and why the coincidence with the published 1–5 cent schedule is treated as suggestive
  rather than confirmatory.
- **Vendor data quality is the binding constraint.** Yahoo's adjusted close is
  dividend-adjusted for most venues but **split-only for the London Stock Exchange**. Five of
  the fifteen pairs in this study die on that alone, and a naive run reports a −5.4 %/yr
  "custody fee" for them. This is not a subtlety in the literature; it is the single largest
  effect in the study, and any replication that skips the coverage screen will publish it as
  a finding.

## Related desk studies (dedup)

- **[Study 955 — ADR Catch-Up](../../955-adr-overnight-catchup/)**: the *high-frequency*
  ADR question — does the ADR still owe you the move the home market already made overnight?
  That is a tradable timing claim on the ratio's *deviations*; 956 deliberately ignores the
  deviations and measures only their **slow drift**, which is the fee.
- **[Study 916 — Withholding Drag](../../916-withholding-drag-international/)**: the same
  tax leak measured *inside a fund wrapper* (VEA and friends), where it was found not to be
  identified on the tape. 956 attacks it at the **single-security** level with the *home
  line itself* as the benchmark rather than a fund blend — a much sharper ruler — and reaches
  the same conclusion on the tax (not identified on the tape) while putting a weakly
  significant *upper bound* on the wrapper's combined income leak, which 916 could not do.
- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  the drift of a *tracker* from its index; 956 is the drift of a *depositary receipt* from its
  underlying, a different wrapper with a different (and published) fee.
- **[Study 889 — Broad Dollar-Hedge Overlay](../../889-dollar-hedge-overlay/)** and
  **[906 — EM Local Bonds FX-Hedged](../../906-em-local-hedged/)**: FX enters as a *risk* to
  strip. Here FX cancels out of the headline estimator entirely (it multiplies a leg's
  total-return and price-only closes identically), which is the estimator's main advantage.
- **[Study 636 — Exchange-Listing Pop](../../636-exchange-listing-pop/)**: an event study on
  the act of listing; 956 measures the standing cost of *being* listed through a depositary.
- **[Study 946 — Distribution ≠ Return](../../946-distribution-rate-illusion/)** and
  **[Study 599 — Tax-Loss Harvesting](../../599-tax-loss-harvesting/)**: distributions and
  taxes as an investor's problem, not as a wrapper-level leak measured against a twin.

## Method lineage

- **HAC / Newey-West standard errors on a trend.** Newey & West (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, Econometrica — [`strategy.nw_ols`](../adr_drag/strategy.py) with 252 lags, chosen
  long because the regression error (the arbitrage band) is strongly persistent, and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_drag_ci`](../adr_drag/strategy.py) resamples blocks of daily changes
  *inside* each break segment and refits, which is the honest counterweight to the HAC *t*
  and here disagrees with it on seven of ten names.
- **Structural break detection by level shift.** Bai & Perron (1998, 2003) on multiple
  structural changes; the detector used here is the cheap non-parametric cousin — a forward
  minus backward rolling median — which is what distinguishes a *permanent* ADS-ratio step
  from a fat-tailed non-synchronous close.
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — as-of slicing
  and a content fingerprint on the exact input panel.

## Data sources

- **Ten ADR / home-line pairs** across France (TTE, SNY), Germany (SAP), the Netherlands
  (PHG, ING), Italy (E), Switzerland (NVS), Denmark (NVO), Japan (TM) and Taiwan (TSM), plus
  **five London pairs** (SHEL, BP, HSBC, UL, RIO) carried through the same pipeline and
  dropped by the coverage screen. Daily closes via `yfinance`, **total return**
  (`auto_adjust=True`) and **price only** (`auto_adjust=False`), 2000-01-03 → 2026-06-30.
- **FX**: `EURUSD=X`, `CHF=X`, `DKK=X`, `JPY=X`, `TWD=X`, `GBPUSD=X` (the last three quoted
  local-per-USD and inverted in the loader). **Cash leg**: `BIL`, for the excess-of-cash
  race in the tradability section only.
- **Non-tape inputs, all labelled and swept**: the treaty withholding rate per country, the
  FX-conversion cost of buying a home line (0–50 bp one-way), and the ongoing foreign
  safekeeping charge (0–30 bp/yr). The published 1–5 cent per ADS fee schedule is used as an
  external benchmark for the result, never as an input.
- **Survivorship.** The ten kept names are large, still-listed issuers with a live ADR and a
  live home line in 2026 — an explicit **survivor panel**. Depositary fees do not plausibly
  depend on survival, but a delisted or de-sponsored ADR (where fee disputes concentrate) is
  by construction absent, so the estimate is a lower bound on the cross-section's tail.
- **As-of 2026-06-30**; the partial current month is dropped so the sample never creeps.
