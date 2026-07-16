# References & literature map — Study 780 (Long-Weekend-Drift)

## The claim under test

- **The folklore.** "Buy the day before a long weekend — the market always drifts up into a
  holiday." The **pre-holiday effect** is one of the oldest documented calendar anomalies:
  the last trading session before a market holiday historically earned a return many times
  the ordinary daily average, with an unusually high fraction of up-days.
- **Why it's a clean calendar test.** The NYSE holiday schedule is **published years ahead**,
  so a "buy K sessions before the holiday, sell on the pre-holiday close" rule is
  calendar-known and zero-look-ahead by construction. The holiday dates are hardcoded from
  the NYSE schedule ([`data.py`](../long_weekend_drift/data.py)); the traded *session* is the
  last SPY close strictly before each holiday, resolved from the tape so half-days and
  weekend-observance shifts handle themselves.
- **The efficient-markets prior + the decay story.** A calendar regularity everyone can put
  in their diary is exactly what a semi-strong-efficient market should arbitrage away — and
  the empirical record is that most published calendar anomalies *shrink or vanish after
  publication* (Schwert 2003; McLean & Pontiff 2016). The desk's prior is a faded effect.

## What the literature actually says about the pre-holiday effect

- **Fields (1934, *Journal of Business*)** — an early note on the pre-holiday tendency of the
  Dow, one of the first documented seasonalities.
- **Ariel (1990, *Journal of Finance*, "High Stock Returns before Holidays")** — the canonical
  modern study: the ~8 pre-holiday sessions a year earned a large share of the total annual
  return in 1963–1982 US data; the effect was economically large and pervasive across size.
- **Lakonishok & Smidt (1988, *Review of Financial Studies*)** — 90 years of the Dow; the
  pre-holiday day is among the most robust of the day-of-week / turn-of-period regularities
  they catalogue, with returns an order of magnitude above the average day.
- **Kim & Park (1994, *JFQA*)** — the pre-holiday effect appears in NYSE, AMEX and NASDAQ and
  is not explained by the other known seasonals; international evidence in **Cadsby & Ratner
  (1992, *Journal of Banking & Finance*)**.
- **Decay after discovery** — **Schwert (2003, *Handbook of the Economics of Finance*)** and
  **Marquering, Nisser & Valla (2006, *Applied Financial Economics*)** document that the
  pre-holiday and several sibling calendar effects **weakened or disappeared** after they were
  published — the natural null for a modern (2005→2025) sample.

## What the literature says about calendar anomalies more broadly

- **McLean & Pontiff (2016, *Journal of Finance*, "Does Academic Research Destroy Return
  Predictability?")** — published anomalies lose ~58% of their return out-of-sample; the
  frame for reading any surviving pre-holiday tilt as *weak echo* rather than *live edge*.
- **Sullivan, Timmermann & White (2001, *Journal of Econometrics*)** — calendar effects are
  especially vulnerable to **data-snooping**; multiple-window testing (we test pre1, pre3,
  post1) inflates apparent significance, so a bootstrap/placebo is mandatory.

## Data & method

- **Real tape:** `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance). Single series, **self-benchmarked**:
  "abnormal" = excess over SPY's own mean daily return (there is no cross-sectional leg).
- **Statistics:** one-sample *t* of the excess return across independent, non-overlapping
  holiday events (the correct unit — not a daily panel, which would fake precision); Wilson
  hit-rate interval; a 20-seed × 200-draw random-window placebo per cut; a leave-one-out
  jackknife; an old/recent sub-sample split; a costed net leg.
- **Synthetic positive control:** a seeded single-series world with a *planted* pre-holiday
  bump — the detector must recover the bump monotonically and stay quiet on the null. See
  [`strategy.py`](../long_weekend_drift/strategy.py).

*Fields, M. (1934). **J. Business**. · Ariel, R. (1990). **J. Finance**. · Lakonishok, J. &
Smidt, S. (1988). **RFS**. · Kim, C. & Park, J. (1994). **JFQA**. · Cadsby, C. & Ratner, M.
(1992). **JBF**. · Schwert, G.W. (2003). **Handbook of the Economics of Finance**. ·
Marquering, W., Nisser, J. & Valla, T. (2006). **Applied Financial Economics**. · McLean, R.D.
& Pontiff, J. (2016). **J. Finance**. · Sullivan, R., Timmermann, A. & White, H. (2001).
**J. Econometrics**.*
