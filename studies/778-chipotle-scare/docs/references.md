# References & literature map — Study 778 (Chipotle-Scare)

## The claim under test

- **The folklore.** "Buy the dip on a Chipotle scare." When a food-safety headline breaks
  — E. coli, norovirus, Salmonella, Clostridium — CMG gaps down on the fear, and a
  contrarian is supposed to buy the panic and ride the recovery, on the theory that a
  burrito chain's brand is durable and the sell-off is an over-reaction. Because the news
  is **already public** when you act, a "buy at the announcement close, hold K sessions"
  rule is executable and zero-look-ahead by construction.
- **Why it is a clean single-name event study.** Each scare is anchored on the trading day
  it became public, market-moving news, hardcoded in [`data.py`](../chipotle_scare/data.py)
  from primary health-authority releases and contemporaneous financial coverage. The
  Aug-2015 Simi Valley, CA norovirus outbreak is deliberately **excluded** — Chipotle did
  not disclose it contemporaneously (it surfaced only in a later lawsuit), so there is no
  honest market-moving date to anchor on.

## The real events (primary + contemporaneous sources)

- **2015-09-17 — Minnesota Salmonella Newport (tomatoes, 64 sick).** The Minnesota
  Department of Health publicly linked the outbreak to Chipotle. (MPR News, "Tomatoes source
  of Chipotle salmonella outbreak in Minnesota," 2015-09-16; *Food Safety News*, Sept 2015.)
- **2015-10-30/31 — E. coli O26, WA/OR (43 restaurants closed).** Chipotle voluntarily
  closed 43 Pacific-Northwest locations; the CDC declared a multi-state outbreak. (CDC,
  "E. coli O26 Infections Linked to Chipotle Mexican Grill Restaurants," Nov 2015; CNN,
  2015-11-01.) Anchored on the first full trading day, 2015-11-02.
- **2015-12-04 — second, genetically distinct E. coli outbreak.** The CDC announced a
  separate STEC O26 outbreak across additional states. (CDC final update, Nov/Dec 2015;
  *Food Safety News*, "A Timeline of Chipotle's Five Outbreaks," Dec 2015.)
- **2015-12-08 — Boston College norovirus (~140 sick).** Including members of the BC men's
  basketball team; the Cleveland-Circle location closed. (Contemporaneous Boston coverage,
  Dec 2015; *Food Safety News*, Dec 2015.)
- **2017-07-18 — Sterling, VA norovirus (~135 sick).** CMG fell ~4% intraday on the report;
  a sick employee was later blamed. (CNBC, "Chipotle shares plummet following report of
  norovirus at Virginia restaurant," 2017-07-18; US News, 2017-07-26.)
- **2018-07-31/08-01 — Powell, OH Clostridium perfringens (~647 sick).** CMG fell ~4.5%;
  the Ohio agency later identified the culprit. (CNBC, 2018-08-03; *Live Science*, 2018;
  QSR Magazine, 2018.) Anchored on 2018-08-01. The 2015 + 2018 outbreaks later drew a
  US-DOJ deferred-prosecution agreement and a $25M fine (DOJ, April 2020).

## What the academic literature actually says

- **Event-study method.** Ball & Brown (1968, *JAR*); MacKinlay (1997, "Event Studies in
  Economics and Finance," *Journal of Economic Literature*) — the canonical framework for
  measuring an asset's *abnormal* return around a discrete corporate event, exactly the
  CMG − SPY construction used here.
- **Reaction to product-harm / recall news.** Jarrell & Peltzman (1985, *JPE*, "The Impact
  of Product Recalls on the Wealth of Sellers") and the marketing product-harm-crisis
  literature (e.g. Van Heerde, Helsen & Dekimpe, 2007, *Marketing Science*) find recalls
  and safety crises impose real, often persistent, shareholder losses — consistent with the
  "keep-falling" pattern we observe, not a quick rebound.
- **The disposition / "catch a falling knife" reflex.** Odean (1998, *JF*) and Barber &
  Odean (2008, *RFS*) document retail investors' attraction to beaten-down, attention-grabbing
  names — motivating *why* the "buy the dip" reflex exists — while offering no evidence it is
  a profitable edge.
- **Efficient-markets prior.** Fama (1970, *JF*, "Efficient Capital Markets") — public,
  widely-reported news should be impounded quickly; a durable post-announcement drift would
  be the anomaly, and with n = 6 (one dominant episode) we cannot claim one.

## Data & method

- **Real tape:** `CMG` and `SPY` daily adjusted (total-return, split-adjusted) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel (2013→2026).
- **Statistics:** one-sample *t* of the abnormal return across scare events; Wilson hit-rate
  interval; a 20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife; a
  costed net leg — with the n = 6 / overlapping-2015-windows dependence flagged throughout.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  acute dip and an optional post-event rebound — the detector must recover a planted rebound
  and stay quiet on the null. See [`strategy.py`](../chipotle_scare/strategy.py).

*Ball, R. & Brown, P. (1968). **JAR**. · MacKinlay, A.C. (1997). **JEL**. · Jarrell, G. &
Peltzman, S. (1985). **JPE**. · Van Heerde, H., Helsen, K. & Dekimpe, M. (2007).
**Marketing Science**. · Odean, T. (1998). **JF**. · Barber, B. & Odean, T. (2008).
**RFS**. · Fama, E. (1970). **JF**.*
