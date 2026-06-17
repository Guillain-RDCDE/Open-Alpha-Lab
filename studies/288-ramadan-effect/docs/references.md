# References & literature map — Study 288 (Ramadan-Effect)

## The claim under test

**Bialkowski, J., Etebari, A. & Wisniewski, T. P. (2012).** "Fast profits:
Investor sentiment and stock returns during Ramadan." *Journal of Banking &
Finance*, 36(3), 835–845.
The canonical academic treatment. Studying 14 predominantly Muslim countries over
1989–2007, the authors report that average stock returns during Ramadan are about
**nine times higher** than the rest of the year, with **lower volatility**. They
attribute this to a sentiment channel: communal fasting fosters optimism, social
cohesion and a sense of well-being, while trading activity thins, easing selling
pressure. This is one of the few calendar anomalies with an explicit behavioural
mechanism rather than a pure data-mining artifact.

## Corroborating and dissenting evidence

- **Al-Hajieh, H., Redhead, K. & Rodgers, T. (2011).** "Investor sentiment and
  calendar anomaly effects: A case study of the impact of Ramadan on Islamic
  Middle Eastern markets." *Research in International Business and Finance*, 25(3),
  345–356. Finds positive Ramadan effects in most MENA markets studied, but with
  considerable cross-country heterogeneity — some markets show no effect.

- **Białkowski, Bohl, Kaufmann & Wisniewski (2013).** "Do mutual fund managers
  exploit the Ramadan anomaly? Evidence from Turkey." *Emerging Markets Review*,
  15, 211–232. Even where the effect exists in the index, fund managers do not
  reliably capture it — a hint that the tradable, net-of-cost edge is thin.

- **Seyyed, Abraham & Al-Hajji (2005).** "Seasonality in stock returns and
  volatility: The Ramadan effect." *Research in International Business and
  Finance*, 19(3), 374–383. Saudi-market study: finds a systematic *decline in
  volatility* during Ramadan but no robust return premium — i.e. the H2 (calmer)
  claim survives better than the H1 (higher) claim.

## Why a desk should be sceptical

- **Tiny effective n.** Ramadan is one month a year. A clean, buyable, US-listed
  single-MENA proxy (iShares MSCI Saudi Arabia, `KSA`) only begins in 2015, giving
  ~11 Ramadan months. With ~5% monthly volatility, the minimum detectable
  mean-difference at 80% power is ~4.7%/mo — larger than the observed gap. The
  literature's significance came from pooling many country-years.

- **Out-of-sample decay.** The original sample ended in 2007. Calendar anomalies
  with behavioural rationales are notorious for fading after publication as the
  trade gets crowded or the regime changes (cf. Harvey, Liu & Zhu 2016 on the
  inflated t-stat hurdle for "new" anomalies, now ~3.0).

- **Placebo discipline.** A genuine behavioural effect must be *absent* on a
  non-Muslim-majority benchmark. Running the identical Ramadan-vs-rest split on
  the S&P 500 is the cheapest falsification test; a positive placebo would brand
  the MENA result a lunar-calendar coincidence.

- **No clean vehicle.** Even a real index effect must survive the opportunity cost
  of being out of the market 11 months a year, plus entry/exit costs and the
  representativeness gap between a single ETF and "MENA equities."

## Method lineage

- **HAC / Newey-West standard errors.** Newey, W. K. & West, K. D. (1987),
  *Econometrica* 55(3). We regress monthly returns on a Ramadan dummy and read the
  HAC (Bartlett kernel, 3 lags) t-stat on the slope — the desk's ≥ 2 inference bar.
- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` on Ramadan vs rest —
  appropriate because the premise itself is that the two groups have different
  variances.
- **Permutation test.** Shuffle the Ramadan labels 10,000 times; the p-value is the
  fraction of shuffles whose absolute mean-gap equals or exceeds the observed one.
- **Power / minimum detectable effect.** Two-sample t MDE at 80% power, α = 0.05,
  to quantify what n = 11 can and cannot resolve.

## Data sources

- **iShares MSCI Saudi Arabia ETF (`KSA`).** The most liquid US-listed single-MENA
  ETF; monthly auto-adjusted closes via yfinance from its 2015 inception. A coarse,
  survivorship-tinged proxy for "MENA equities."
- **S&P 500 (`^GSPC`).** Non-Muslim-majority placebo tape, same window.
- **Ramadan windows.** Hardcoded in `data.py` from the Umm al-Qura (Saudi civil)
  calendar, the standard reference for the Gulf markets that dominate the proxy.
  Sighting-based observance can differ by a day, immaterial at monthly resolution.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: a folklore calendar/event
  indicator with the same tiny-n, base-rate-trap structure.
- Halloween / "Sell in May" and the January effect are in the same seasonal family
  — known-in-advance calendar windows whose edges shrink under honest inference.
