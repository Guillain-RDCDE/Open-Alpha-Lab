# References & literature map — Study 797 (FX Value / PPP)

## The claim under test

- **The folklore / practitioner claim.** "**Cheap currencies bounce.**" A currency whose
  *real* exchange rate — nominal FX adjusted for relative inflation — sits below its own
  long-run average is "undervalued" versus purchasing-power parity (PPP) and is expected
  to appreciate back toward fair value; an "expensive" (overvalued) currency is expected
  to give it back. Ranking currencies on that PPP gap and going **long the cheap / short
  the rich** is the FX **value** factor.
- **The academic anchor.** Absolute and relative PPP and the slow mean reversion of the
  real exchange rate are foundational international-finance results — Rogoff (1996,
  *The Purchasing Power Parity Puzzle*, JEL) documents the famous ~3–5-year half-life of
  real-rate deviations (slow, but real). The *cross-sectional* value trade is formalised
  by **Asness, Moskowitz & Pedersen (2013, "Value and Momentum Everywhere", JF)**, who
  use the 5-year change in the real exchange rate as the currency value signal, and by
  **Menkhoff, Sarno, Schmeling & Schrimpf (2017, "Currency Value", CEPR/RFS)**, who build
  the real-exchange-rate deviation from PPP and show a long-cheap/short-rich portfolio
  earns a premium — **concentrated in, and strongest for, emerging-market currencies**,
  weaker within the developed G10 alone.
- **The open question we test.** Does the PPP-value tilt clear the desk's `t ≥ 2` bar on
  a **developed-market G10-only** tape, once built point-in-time (CPI publication lag)
  and charged realistic costs and short borrow? (Spoiler in [`results.md`](results.md):
  directionally yes, statistically `WEAK`, and a `MIRAGE` net of frictions — exactly the
  developed-market-is-weaker caveat in Menkhoff et al.)

## What we measure, and the honesty rails

- **Real rate.** `log q_i = log S_i(USD/FX) + log CPI_i − log CPI_US`. High q = the
  currency is *rich* (overvalued) vs PPP; low q = *cheap* (undervalued). The value signal
  is `trailing-mean(log q, 60 months) − log q_t` — positive when the real rate is below
  its own long average. Cross-sectional dollar-neutral rank weights, long cheap / short
  rich, rebalanced monthly.
- **Point-in-time, one documented lag.** CPI for month *m* is released weeks into month
  *m+1*, so the signal at the close of month *t* uses CPI known only through *t−1* (a
  1-month publication lag) and earns the month *t+1* spot return — a single `shift`, no
  look-ahead into an unreleased print.
- **Inference.** One-sample *t* and a **Newey-West (1987)** HAC *t* on the monthly
  long-short return (the decisive number), a **random-sign placebo** (10,000 books), a
  **Wilson (1927)** interval on the monthly hit rate, and a trailing-window robustness
  sweep. A post-2015 sub-period result is reported but flagged as **snooped** (not
  pre-registered) and excluded from the stamp.
- **Costs & borrow.** Turnover = one-way × NAV of the monthly weight change (both legs);
  **shorts pay a 100 bps/yr borrow/financing spread** on gross short notional. Costs
  charged against the alpha, not the gross.
- **Named data coarseness.** Australia and New Zealand publish CPI only **quarterly**; we
  forward-fill their real rate within each quarter, a documented approximation on 2 of 10
  legs. Euro-area inflation is HICP (not the OECD CPI used for the others) — the
  standard euro-wide price index.

## Data sources

- **G10 FX spot** — yfinance (no key), month-end, USD-per-foreign, cached under `_cache/`.
- **CPI** — DBnomics mirrors of IMF IFS (`PCPI_IX`) and Eurostat HICP (`prc_hicp_midx`),
  cached under `_cache/`.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [215-big-mac-ppp](../215-big-mac-ppp/) — a single **Big-Mac folklore snapshot**: one
  cross-section of a burger-basket PPP gap, no real-rate *time series* and no
  point-in-time backtest. This study builds the **CPI-based real-rate history** and
  actually trades the deviation month by month.
- [364-fx-carry-trade](../364-fx-carry-trade/) — the **opposite tilt**: rank on the
  *interest-rate differential* and go long high-yielders. Carry and value are famously
  **negatively correlated** styles; this study is the value leg, not carry.
- [147-fx-momentum](../147-fx-momentum/) — ranks on recent **price trend**, not on a
  PPP/real-rate level. Value is the mean-reversion counterpart to momentum's trend.
- [114-dollar-smile](../114-dollar-smile/) — the **broad-dollar cycle / risk-regime**
  level (when does the USD rally?), a single-factor macro view, not a cross-sectional
  cheap-vs-rich sort of the other nine currencies.

None of the siblings build the **real-exchange-rate deviation-from-PPP time series** and
trade it long-cheap/short-rich — that is this study's own axis.
