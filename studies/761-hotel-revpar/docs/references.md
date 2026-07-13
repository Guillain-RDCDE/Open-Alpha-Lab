# References & literature map — Study 761 (Hotel-RevPAR)

## The claim under test

- **RevPAR as the lodging cycle's headline gauge.** **STR** (Smith Travel Research, now
  part of **CoStar Group**) is the hotel industry's standard benchmarking source; its
  monthly **RevPAR** (Revenue Per Available Room = ADR × occupancy) is the single number
  hotel operators, REITs and analysts track. The **American Hotel & Lodging Association
  (AHLA)** "State of the Industry" reports and STR's monthly press releases publish the
  headline U.S. figures. RevPAR growth is treated as *the* read on travel demand.
- **The folklore.** "RevPAR momentum leads hotel stocks." When RevPAR is accelerating —
  occupancies and room rates beating last year — the travel cycle is turning up, so be long
  hotel REITs; the demand strength isn't fully priced yet. RevPAR YoY is routinely charted
  against lodging-REIT performance as a leading/coincident risk-on signal for the sector.

## Why we construct a proxy — and how

- **STR's monthly series is proprietary.** The licensed STR/CoStar monthly RevPAR tape
  (and its ADR/occupancy components, by chain scale and market) is commercial data behind a
  paywall; there is **no free monthly history** on FRED or yfinance. This is exactly the
  situation Study 358 (Watch-Index) and Study 708 (Eurovision) face, and we resolve it the
  same way: a **small, clearly-labelled approximate reconstruction**, never presented under
  a real-tape banner.
- **The transparent proxy.** We hardcode a monthly U.S. RevPAR path (dollars) **anchored to
  STR/CoStar-reported annual RevPAR** (widely republished in AHLA/STR year-in-review
  material), shaped by realistic hotel seasonality (summer peak, winter trough), with the
  **2020 COVID collapse** (national RevPAR fell below ~$21 in April 2020) and the **2021
  recovery** set to the reported national monthly figures. The signal is RevPAR **YoY log
  momentum**, which is invariant to the seasonal shape and to any constant scaling — so the
  study rides on the travel cycle's *shape and turning points*, which the anchors pin down,
  not on any single month's exact dollar figure. Labelled a **PROXY** throughout.

## Why a coincident demand gauge need not lead the stocks — the economics & statistics

- **Equities are forward-looking claims.** A hotel REIT's price is the discounted value of
  *future* room revenue, so it should move on *expected* RevPAR, ahead of the realized
  print — the classic result that stock prices lead the macro/fundamental series they
  reflect (Fama, 1981, *Stock Returns, Real Activity, Inflation, and Money*, AER; Stock &
  Watson, 2003, on asset prices as leading indicators). Our lead-lag test operationalises
  exactly this: the equity leading the gauge.
- **Reporting & release lag.** STR's monthly RevPAR is compiled and released ~2–3 weeks
  after month-end, so even the *timely* read reaches a trader weeks after the demand it
  measures — and long after the equity has repriced. We impose a one-month release lag and
  document it.
- **Late-cycle / mean-reversion.** A cyclical demand series at its YoY peak tends to precede
  *lower* forward asset returns (high growth = late cycle), the sectoral analogue of the
  dividend/earnings-growth "expectations" literature (Lakonishok, Shleifer & Vishny, 1994,
  *Contrarian Investment, Extrapolation, and Risk*, JF). We treat the significant *negative*
  slope cautiously: with overlapping windows and two cyclical series it is partly mechanical.
- **Small-sample / base-rate inference.** Equities rise in most windows, so a high
  post-signal win-rate is expected under the null; the honest object is the **excess** over
  the base rate (Kahneman & Tversky, 1973), tested with a **Welch two-sample t** (Welch,
  1947) and a **placebo / randomization** null (Fisher's logic; Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993).
- **Overlapping returns ⇒ HAC.** Regressing overlapping H-month forward returns on the
  signal induces strong serial correlation; ordinary SEs are far too small. We use
  **Newey-West** (Newey & West, 1987) heteroskedasticity-and-autocorrelation-consistent
  standard errors at the overlap length — the desk's |t| ≥ 2 bar.

## Method lineage (the desk's shared engine)

- **YoY momentum + conditional split.**
  [`strategy.revpar_momentum`](../hotel_revpar/strategy.py) and
  [`strategy.split_returns`](../hotel_revpar/strategy.py) — UPCYCLE vs DOWNCYCLE vs
  unconditional forward returns.
- **HAC predictive regression + Welch t + placebo p.**
  [`strategy.newey_west_t`](../hotel_revpar/strategy.py),
  [`strategy.predictive_regression`](../hotel_revpar/strategy.py),
  [`strategy.welch_t`](../hotel_revpar/strategy.py),
  [`strategy.placebo_pvalue`](../hotel_revpar/strategy.py).
- **Lead-lag cross-correlation.** [`strategy.lead_lag`](../hotel_revpar/strategy.py) — the
  direction-of-causation test (does the gauge lead the tape, or the reverse?).
- **Timing backtest, net of costs.**
  [`strategy.timing_backtest`](../hotel_revpar/strategy.py) — long/flat (or long/short)
  hotel tape held when RevPAR YoY > 0, one-month lag, one-way cost per turn, raced against
  buy-and-hold on a Sharpe basis (total-return, labelled).
- **Deterministic synthetic control.** [`data.synthetic`](../hotel_revpar/data.py) plants a
  known forward lead tied to RevPAR momentum; the offline core runs with no network. The
  control confirms the inference recovers a planted lead **and** does not manufacture
  significance when the true edge is zero.

## Data sources used here

- **RevPAR proxy:** hardcoded monthly table in
  [`hotel_revpar/data.py`](../hotel_revpar/data.py), anchored to STR/CoStar-reported annual
  U.S. RevPAR. **yfinance** daily total-return closes for **HST** (cached under
  `_cache/hst.csv`) and the equal-weight lodging-REIT basket HST/RHP/SHO/DRH/PEB/APLE/PK
  (`_cache/basket.csv`), resampled to month-end. Window 1998-01 → 2026-05 (28.3 years). All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Macro-nowcasting & leading-indicator gauges** on this bench — the ISM-PMI regime,
  jobless-claims momentum, the economic-surprise index — share the lesson: a real
  fundamental relationship rarely survives as a *tradable* monthly timing rule, and a gauge
  the equity already discounts is a coincident-or-lagging tell, not the early read the
  folklore sells.
