# References & literature map — Study 750 (Return-to-Office)

## The claim under test

- **The folklore.** Every time a marquee employer orders workers **back to the office**,
  the narrative goes, office demand firms up and the beaten-down office landlords should
  catch a bid. The mandates are vivid and dated: **Goldman Sachs'** David Solomon called WFH
  "an aberration" (Feb 2021); **Elon Musk's** "40 hours in the office or leave" email at
  Tesla (Jun 2022) and his end of remote work at Twitter/X (Nov 2022); **Disney's** Bob Iger
  4-day mandate (Jan 2023); **Amazon's** 3-day (Feb 2023) then full **5-day** RTO
  (Sep 2024); **JPMorgan's** all-employees-5-days order (Jan 2025); and the **US federal
  government** RTO executive order (Jan 2025). Bulls on office REITs (SL Green, Boston
  Properties, Vornado…) point to exactly these headlines.
- **Where it's repeated.** Financial media and sell-side notes routinely tie office-REIT
  moves to RTO headlines (e.g. Bloomberg, CNBC, WSJ coverage of SL Green rallies "as workers
  return"). The believers' framing is that a *stricter* mandate (full 5-day) should move
  offices more than a soft hybrid — we test that split explicitly.

## Why the reaction is unlikely to survive a clean test

- **Offices are a rates + structural-vacancy trade.** Office-REIT valuations in 2021–2025
  were dominated by the interest-rate cycle and by **secular** work-from-home vacancy, not by
  any single employer's memo. On record office availability, see **Cushman & Wakefield** and
  **CBRE** quarterly office MarketBeat reports (US office vacancy hit multi-decade highs,
  ~19–20%, 2023–2024). The macro literature on remote work: **Barrero, Bloom & Davis (2021),
  *Why Working From Home Will Stick*, NBER WP 28731** (WFH settled well above pre-COVID and is
  sticky); **Gupta, Mittal, Peeters & Van Nieuwerburgh (2022), *Flattening the Curve: Pandemic-
  Induced Revaluation of Urban Real Estate*, Journal of Financial Economics** — a long-run
  **~$500bn** repricing of NYC office value, driven by structural WFH, not by RTO news.
- **Anticipation & efficient pricing.** By the time a mandate is announced it has usually been
  trailed for weeks, so an efficient market has little to reprice on the day (Fama, 1970,
  *Efficient Capital Markets*, JF). A short-window event study is precisely the instrument to
  detect any residual reaction — and to certify its absence.
- **Small-sample inference.** With ~26 dated mandates on a **single** basket, the
  cross-section of event CARs has a large standard error. We test the mean against zero with a
  **one-sample t** (Welch, 1947) and, because the sample is tiny, with a **placebo /
  randomisation null** — random non-event windows of the *same basket* — to size the
  small-sample sampling distribution (Fisher's randomisation logic; Efron & Tibshirani, 1993,
  *An Introduction to the Bootstrap*).

## The event-study method (the desk's shared engine)

- **Market-model CAR.** [`strategy.event_car`](../return_to_office/strategy.py) fits
  `r_basket = α + β·SPY` on a clean 120-day pre-event estimation window (10-day gap, no
  leakage) and cumulates the abnormal return over the event window — the canonical
  short-window design of **MacKinlay (1997), *Event Studies in Economics and Finance*, JEL**,
  and **Brown & Warner (1985), *Using Daily Stock Returns*, JFE**.
- **Basket construction.** [`strategy.basket_returns`](../return_to_office/strategy.py) —
  equal-weight daily returns of the surviving office REITs. A **VNQ** (broad-REIT) benchmark
  pass asks whether office reacts *beyond* the whole REIT complex.
- **Placebo null + Welch t.** [`strategy.placebo_car_dist`](../return_to_office/strategy.py)
  and [`strategy.welch_t`](../return_to_office/strategy.py) — the all-events CAR vs zero, the
  strict−hybrid difference, and a 20,000-draw randomisation null sized to the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../return_to_office/data.py) plants a known strict-bucket CAR edge
  of size `car_bps`; with `car_bps=0` the inference must NOT manufacture a strict>hybrid gap,
  and a large edge must light up. Runs offline.
- **Costs.** [`strategy.net_of_costs`](../return_to_office/strategy.py) charges a one-way
  round-trip on a "buy the basket on the mandate, hold the window" trade.

## Survivorship — named on the Signal axis

- The priced basket is the office landlords that **survived** the WFH shock. The worst
  casualties left the tape: **WeWork** (WE) went bankrupt and delisted (Nov 2023); scores of
  private office towers were surrendered to lenders via **CMBS default** through 2023–2024;
  distressed office REITs (Office Properties Income Trust, OPI) did dilutive debt exchanges and
  reverse splits. Listed in [`data.DELISTED`](../return_to_office/data.py). The bias points
  **against** the office distress the RTO story is meant to reverse, so a survivor basket that
  fails to pop is a *conservative* refutation (Brown, Goetzmann, Ibbotson & Ross, 1992,
  *Survivorship Bias in Performance Studies*, RFS).

## Data sources used here

- **yfinance** daily adjusted closes for 10 office REITs + SPY + VNQ, 2018-06-01 → 2026-07-10,
  cached under `_cache/`. The RTO calendar (date, employer, strict/hybrid) is hardcoded in
  [`data.RTO_EVENTS`](../return_to_office/data.py) from company memos and contemporaneous
  press coverage (WSJ / Reuters / Bloomberg / CNBC / FT).
- **Kastle Systems, "Getting America Back to Work" — Back to Work Barometer** (10-city office
  badge-swipe occupancy vs Feb-2020 = 100). A **cited, approximate, hardcoded PROXY**
  ([`data.KASTLE_OCCUPANCY`](../return_to_office/data.py)) for the *physical* RTO trend, used
  for context only — never priced, never under a real-tape banner.
- Headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the methodological twin — a
  short-window market-model event study on a hardcoded, labelled table, where the only real
  move is the un-tradable announcement instant.
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: the same small-sample /
  survivorship pathology on a table of corporate events (theme-chasing rebrands).
