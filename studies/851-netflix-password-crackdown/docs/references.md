# References & literature map — Study 851 (Netflix Password Crackdown)

## The claim under test

- **The story.** Netflix's **paid-sharing ("password crackdown")** — charging members
  who share an account outside their household — was, through 2022, widely expected to
  drive furious customers away and spike churn. Instead subscriber growth *accelerated*:
  by Q3 2023 Netflix reported its largest quarterly paid net-adds since 2020, and the
  stock re-rated. The narrative became "the scary policy that worked" — a corporate
  turnaround that surprised to the **upside**.
- **The specific test here.** A single-name **news-reaction event study**: NFLX's
  **abnormal returns** (a one-factor market model vs SPY, cross-checked vs QQQ) around
  the **five public market-facing dates** of the paid-sharing saga. We ask whether the
  "upside surprise" showed up as a systematic, *tradable* abnormal return around those
  dates — with the explicit, honest caveat that **five events carry almost no
  statistical power** (this is a case study, not a factor).

## The public-record event calendar (hardcoded in `data.EVENTS`)

Each date is a matter of public record; the reaction session is the first NYSE session
that could trade the news (earnings print after the US close, so the reaction is the
next morning — the study's single documented execution lag):

- **2022-04-19 (react 2022-04-20).** Q1 2022 shareholder letter first flags that Netflix
  will start charging for account sharing, alongside its first subscriber decline in a
  decade — the stock fell ~35% the next session. *Source: Netflix Q1 2022 letter to
  shareholders.*
- **2022-08-22.** Netflix begins testing an "add a home" paid-sharing charge in
  Argentina, the Dominican Republic, El Salvador, Guatemala and Honduras. *Source:
  Netflix Help Center / same-day trade press.*
- **2023-05-23.** Netflix emails US members that sharing outside a household now costs an
  extra $7.99/month — the broad US "password crackdown" rollout. *Source: Netflix
  newsroom, 2023-05-23.*
- **2023-07-19 (react 2023-07-20).** Q2 2023 letter reports +5.9M paid net-adds as the
  crackdown lands, but revenue slightly missed — the stock fell ~8% the next session.
  *Source: Netflix Q2 2023 letter.*
- **2023-10-18 (react 2023-10-19).** Q3 2023 letter reports +8.8M paid net-adds — the
  largest quarterly gain since 2020 — plus a US price rise; the stock rose ~16% the next
  session. *Source: Netflix Q3 2023 letter.*

## What we measure, and the honesty rails

- **Market-model abnormal returns (Brown & Warner 1985; MacKinlay 1997).** The "normal"
  return is a one-factor market model `α + β·r_mkt` whose parameters are OLS-estimated on
  a **120-session estimation window ending 10 sessions before** the event window, so the
  abnormal return over the window is genuinely out-of-sample. `market_adjusted` (β≡1) and
  `mean` (constant-mean) models are provided as cross-checks; the headline is the market
  model.
- **Small-N inference, stated plainly.** With five independent events the cross-event
  one-sample *t* has **4 degrees of freedom** and famously fat tails (|t|≥2 ≈ p 0.12, not
  0.05), so we lean on a **non-parametric random-calendar placebo** (4,000 draws of 5
  random dates), an **event-bootstrap CI**, and a **leave-one-out** cut rather than the
  *t* alone. The synthetic control is deliberately run on **30** pseudo-events so the
  machinery's calibration can be judged free of the small-N fat tails.
- **One documented execution lag.** Earnings react on the session after the after-close
  print; intraday policy announcements react same-day. Encoded in the calendar — zero
  look-ahead.
- **The timer is graded separately.** Costs are one-way × NAV, charged twice per round
  trip; long-only (no borrow) — the honest test of whether any event drift survives
  friction on five trades.

## Shared method citations

- **Brown, S. & Warner, J. (1985)** — "Using daily stock returns: the case of event
  studies" (the market-model / constant-mean abnormal-return framework).
- **MacKinlay, A. C. (1997)** — "Event studies in economics and finance" (estimation
  window, event window, CAR conventions).
- **Wilson, E. B. (1927)** — score interval for a binomial share (the event hit-rate).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return) for NFLX, SPY, QQQ,
  2015-01-02 → 2026-06-30, cached under `_cache/`.
- **Netflix quarterly shareholder letters and newsroom** for the five event dates (public
  record; encoded with per-row source notes in `data.EVENTS`).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [551-netflix-top10](../../551-netflix-top10/) — a **different NFLX signal**: whether the
  weekly Top-10 content chart predicts the stock. This study sorts on **corporate-policy
  event dates**, not content popularity.
- [552-app-store-rankings](../../552-app-store-rankings/) — **app-download / ranking
  alt-data** as a cross-sectional signal. This study is a single-name **event study**
  around named public dates, not an alt-data panel.
- [299-keynote-drift](../../299-keynote-drift/) — drift around **scheduled product
  keynotes** (a recurring, calendar-known announcement type). The crackdown dates here
  are **irregular, one-off policy milestones**, and the question is an *upside-surprise*
  reaction, not pre/post-keynote drift.
- [622-thematic-etf-curse](../../622-thematic-etf-curse/) — the **launch-timing curse** of
  narrative-driven thematic ETFs (a fund-flow / reflexivity story). This study is a
  single stock's reaction to its own policy news, not a fund-launch phenomenon.

None of the siblings run a **market-model event study of NFLX's abnormal returns around
the paid-sharing policy dates** — this study's own axis. All share the desk's single
question: *does a compelling story move a tradable price, after honest inference and
costs?*
