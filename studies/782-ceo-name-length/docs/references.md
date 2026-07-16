# References & literature map — Study 782 (CEO-Name-Length)

## The claim under test

- **The "characteristic."** Does the number of letters in a company's CEO surname predict
  its stock return? This is a **null-by-design** test: surname length carries no economic
  information, so a clean market should show a flat spread. We sort a fixed 40-name large-cap
  universe by ``len(CEO surname)`` and hold a dollar-neutral **long longest-surname / short
  shortest-surname** tercile book, rebalanced monthly. Surnames are real, publicly-verifiable
  facts (company proxy statements / newsrooms), taken as a **static 2026-06 snapshot** —
  CEO turnover through history is deliberately not tracked, a simplification that is only
  defensible *because* the characteristic is inert by construction (see the loud disclosure
  in [`data.py`](../ceo_name_length/data.py)).
- **Why an absurd characteristic is worth running.** It is a live-fire demonstration of the
  data-snooping trap: with hundreds of candidate "signals," some inert label will always
  cross a t-stat bar in-sample by coincidental alignment with a real return driver (here, a
  mega-cap-tech tilt). The value is in showing *how* a nonsense sort manufactures a
  significant-looking number, and how the diagnostics (mechanism, effective sample size,
  confound) expose it.

## What the literature actually says

- **Multiple testing / data snooping in finance.** White (2000, *Econometrica*, "A Reality
  Check for Data Snooping"); Sullivan, Timmermann & White (1999, *JF*) on calendar-effect
  snooping; Harvey, Liu & Zhu (2016, *RFS*, "…and the Cross-Section of Expected Returns")
  argue most published factors don't survive a multiple-testing haircut. A surname-length
  "factor" is the reductio of this literature.
- **Spurious characteristics & the factor zoo.** Cochrane (2011, *JF* presidential address,
  "Discount Rates") on the proliferation of return predictors; Hou, Xue & Zhang (2020, *RFS*,
  "Replicating Anomalies") on how many anomalies vanish out-of-sample or under alternate
  specs. Our result is a textbook in-sample-only spread with no mechanism.
- **Names and markets (the fun corner).** Alter & Oppenheimer (2006, *PNAS*, "Predicting
  short-term stock fluctuations by using processing fluency") found fluently-named tickers
  outperformed shortly after IPO — a genuine, if fragile, *name*-based curiosity; Green &
  Jame (2013, *JFE*, "Company name fluency, investor recognition, and firm value") on
  company-name fluency and liquidity. These concern *fluency/recognition of the firm's own
  name*, not the CEO's surname length, and none imply a tradable surname-length premium.
- **CEO / manager effects.** Bertrand & Schoar (2003, *QJE*, "Managing with Style") show
  individual managers matter for firm policies — but through *style and decisions*, not the
  orthography of their surname.

## Data & method

- **Real tape:** the 40 tickers plus `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), 2015-01 → 2026-06. The long/short book
  is dollar-neutral so it is market-neutral by construction; SPY is the named reference tape
  and a neutrality check (realised LS beta to SPY ≈ −0.31, i.e. not perfectly neutral).
- **Statistics:** one-sample *t* of the monthly LS return; Wilson hit-rate interval; a
  20-seed × 200-draw **label-shuffle placebo** (permute surname lengths across names and
  recompute the LS mean); a leave-one-ticker-out jackknife; alternate quantile cuts; a
  costed net leg. See [`strategy.py`](../ceo_name_length/strategy.py).
- **Synthetic positive control:** a seeded cross-sectional world with a *planted* linear
  characteristic→return slope — the sort must recover a planted slope monotonically and stay
  quiet on the null (bump = 0).

*White, H. (2000). **Econometrica**. · Sullivan, R., Timmermann, A. & White, H. (1999).
**JF**. · Harvey, C., Liu, Y. & Zhu, H. (2016). **RFS**. · Cochrane, J. (2011). **JF**. ·
Hou, K., Xue, C. & Zhang, L. (2020). **RFS**. · Alter, A. & Oppenheimer, D. (2006). **PNAS**.
· Green, T.C. & Jame, R. (2013). **JFE**. · Bertrand, M. & Schoar, A. (2003). **QJE**.*
