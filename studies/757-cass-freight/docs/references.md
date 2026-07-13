# References & literature map — Study 757 (Cass-Freight)

## The claim under test

- **The Cass Freight Index.** Cass Information Systems publishes a monthly **Cass Freight
  Index** with two components — *Shipments* (volume of freight moved) and *Expenditures*
  (dollars spent), base January 1990 ≈ 1.00. It is derived from the ~$44bn of freight
  invoices Cass processes annually for its freight-payment clients across truck, rail,
  intermodal and parcel. Cass's own monthly reports and the accompanying *Cass Transportation
  Index Report* (with ACT Research's Tim Denoyer) explicitly frame the shipments series as a
  **read on the health of the U.S. goods economy** and a signal that "leads" softening or
  recovery. FRED mirrors a version as `FRGSHPUSM649NCIS`.
- **The folklore.** "Trucks and trains move the real economy, so watch the freight." The
  narrative on FreightWaves, in sell-side macro notes and across finance-Twitter: when the
  Cass Freight Index rolls over, an industrial slowdown — and a lower stock market — is
  already in motion but not yet priced, so a freight downturn is a **leading** sell signal
  (and a freight upturn a buy signal), especially for transports (IYT / the Dow Jones
  Transportation Average, the oldest "confirmation" index in Dow Theory).

## Why we hardcode a labelled proxy — and how

- **The real series is not freely reproducible.** Cass's full monthly shipments history sits
  in its reports and licensed feeds; the FRED mirror is not reliably fetchable from this
  environment and its levels/rebasing shift over time. So we **hardcode a small, cited,
  APPROXIMATE annual path** of the shipments level (`cass_freight/data.py::_CASS_ANNUAL`,
  base Jan-1990 ≈ 1.00) and interpolate it to a monthly **LABELLED PROXY** with a fixed
  deterministic seasonal shape. It is a stand-in for the real tape, named a proxy
  everywhere, and the study conditions on its **year-over-year growth** — expansions vs the
  documented contractions (2001–02 dot-com, 2008–09 GFC, 2015–16 industrial/freight
  recession, 2019 slowdown, 2020 COVID air-pocket, 2022–24 freight recession) — a shape the
  exact decimals barely move. The equity tapes (SPY, IYT) are **real** yfinance closes.

## Why a real-economy gauge need not lead the market — the economics & statistics

- **Markets discount the future; freight measures the present.** Equities are a forward-
  looking, near-continuously-priced claim on future cash flows, whereas a freight index is a
  **coincident-to-lagging** measure of goods already shipped, published with a delay. Under
  semi-strong efficiency the industrial cycle a freight print reveals is already in prices —
  so the empirical question is the **sign of the lead-lag**: does freight lead stocks, or do
  stocks lead freight? (The Conference Board classifies the S&P 500 itself as a *leading*
  indicator and industrial/goods activity as *coincident*.)
- **Publication delay kills a naïve overlay.** The Cass reference-month index is released in
  the **middle of the following month**, so a signal is not actionable until ~6 weeks after
  the month it describes — we impose a 2-month lag (publication + execution). A study that
  reads the print at the reference-month close would be trading on look-ahead it never had.
- **Small-sample / base-rate inference.** US equities rise in most rolling windows, so a high
  post-signal win-rate is expected under the null; the right object is the **excess** over
  the unconditional base rate (Kahneman & Tversky, 1973). We test it with a **Welch
  two-sample t** (Welch, 1947) and a **placebo / randomization** null (Fisher's logic; Efron
  & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Lead-lag cross-correlation.** The decisive diagnostic is the cross-correlation of the
  *change* in freight growth with equity returns across leads and lags — a standard tool for
  ordering business-cycle series (Stock & Watson, 1989, *New Indexes of Coincident and
  Leading Economic Indicators*, NBER Macro Annual; Burns & Mitchell, 1946, *Measuring
  Business Cycles*). A genuine leading indicator peaks at a positive lead.
- **Dow Theory & the transports.** The idea that the transportation average must "confirm"
  the industrials (Charles Dow, *Wall Street Journal* editorials, 1900–1902; Rhea, *The Dow
  Theory*, 1932) is the ancestor of "watch the freight." IYT (iShares U.S. Transportation
  ETF, inception 2003) is the tradable sector proxy tested here.

## Method lineage (the desk's shared engine)

- **Welch t + placebo p-value.** [`strategy.welch_t`](../cass_freight/strategy.py) and
  [`strategy.placebo_pvalue`](../cass_freight/strategy.py) — conditional vs unconditional
  forward returns and a 20,000-draw randomization null sized to the event count.
- **Lead-lag cross-correlation.**
  [`strategy.lead_lag_corr`](../cass_freight/strategy.py) — ∆(freight YoY) against equity
  monthly returns across ±12 months; the peak lag orders the two series in time.
- **Timing overlay, net of costs.**
  [`strategy.timing_backtest`](../cass_freight/strategy.py) — long/flat (or long/short) SPY
  or IYT held when freight is expanding, a 2-month publication+execution lag, one-way cost
  per turn, raced against buy-and-hold on a Sharpe basis (price-only, labelled).
- **Deterministic synthetic control.**
  [`data.synthetic`](../cass_freight/data.py) plants a known forward edge tied to freight;
  the offline core runs with no network. The control confirms the inference recovers a
  planted edge **and** does not manufacture significance when the true edge is zero.

## Data sources used here

- **Cass Freight Index (Shipments)** — hardcoded approximate annual anchors interpolated to a
  monthly **proxy** in [`cass_freight/data.py`](../cass_freight/data.py); sourced to Cass's
  monthly reports and the FRED mirror `FRGSHPUSM649NCIS`. **yfinance** month-end adjusted
  closes for **SPY** (1993→) and **IYT** (2004→), cached under `_cache/`. Window
  1999-06 → 2026-06 (27.0 years); IYT tests use the 2004→ sub-window. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Macro leading-indicator & regime gauges** on this bench — the economic-surprise index,
  the ISM-PMI regime, jobless-claims momentum — share the lesson: a real macro relationship
  rarely survives as a *tradable* monthly timing rule once you charge it the publication
  delay and the opportunity cost of sitting in cash through an up-drifting market, and the
  "leading" ones often turn out to *lag* the market that already moved.
