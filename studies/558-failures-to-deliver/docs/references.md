# References & literature map — Study 558 (Failures-To-Deliver)

## The claim, at full strength

- **The meme-stock-era folklore.** Retail message-board and financial-media narratives (2020–2021,
  GME/AMC and beyond) held that a **spike in failures-to-deliver (FTD)** — shares a seller was
  obliged to deliver at settlement but did not — is a **short-squeeze predictor**: persistent fails
  mean shorts are trapped, so a violent upward move is imminent. The tradable version is *buy names
  whose FTD just spiked and ride the pop*. This study tests whether post-FTD-spike forward returns
  are abnormally positive.

## What failures-to-deliver actually are (the data reality)

- **SEC fails-to-deliver data.** Under Regulation SHO the SEC publishes a semi-monthly
  **fails-to-deliver file**: aggregate *open* fail-to-deliver quantity per **CUSIP**, with the
  settlement date, a price field and the security description. It is a bulk flat file, lagged by
  the settlement cycle, keyed by CUSIP (not ticker), and carries **no return/volume panel**
  alongside it. There is *no* free retail-stack endpoint (``yfinance`` has no FTD data), and a
  survivorship-clean CUSIP→ticker→adjusted-price join across the meme-stock cohort is a
  data-engineering project in its own right. Hence this study is **synthetic-only**, and states the
  data-availability limitation openly on the SIGNAL axis (a synthetic-only study can never earn a
  ``REAL`` stamp — that needs a robust *t* ≥ 2 on a real tape).
- **Regulation SHO** (SEC, 2005; amendments 2008–2009). The rule framework — close-out
  requirements, the threshold-securities list, the 2008 emergency naked-short-selling order — that
  makes FTD an object of market-structure attention in the first place.

## Why the folklore is fragile — the microstructure literature

- **Boni (2006)**, *"Strategic delivery failures in U.S. equity markets."* *Journal of Financial
  Markets* 9(1). Documents that fails are common, persistent, and often reflect ordinary
  market-making and mechanical settlement frictions rather than a cornered short — undercutting the
  "trapped shorts → squeeze" reading.
- **Evans, Geczy, Musto & Reed (2009)**, *"Failure Is an Option: Impediments to Short Selling and
  Options Prices."* *Review of Financial Studies* 22(5). Option market-makers *rationally* fail to
  deliver when borrow is expensive — fails are a *cost-of-borrow* phenomenon, not a squeeze signal.
- **Fotak, Raman & Yadav (2014)**, *"Fails-to-deliver, short selling, and market quality."*
  *Journal of Financial Economics* 114(3). Finds fails are, if anything, associated with *better*
  liquidity and *lower* volatility on average — the opposite of the impending-squeeze story.
- **Culp & Heaton (2008)** and subsequent SEC economic notes — the case that fails are largely a
  settlement-plumbing artifact, dominated by legitimate market-making, and a poor cornering proxy.

## The event-study method we build

- **Event-study abnormal returns.** We test the folklore as a classic event study: flag FTD-spike
  events (a trailing-z-score threshold, causal), compute the forward cumulative return over a
  short horizon, and **market-adjust** (subtract the cross-name forward return) to isolate the
  *abnormal* post-spike return — the pop *beyond* ordinary drift, which is what the claim asserts.
  See MacKinlay (1997), *"Event Studies in Economics and Finance"*, *Journal of Economic
  Literature* 35(1), for the abnormal-return / CAR framework.
- **Non-overlapping events (refractory period).** Because FTD is persistent, raw spike days sit
  inside each other's forward windows; overlapping event returns are dependent and would inflate a
  naive *t*. We thin spikes to a refractory spacing of one horizon so event windows are disjoint
  and the sample is approximately independent (Brown & Warner 1985 on the perils of clustered /
  overlapping event windows).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: move the
  spike labels to random non-event days and read the abnormal-return tail probability.
- **Multiple-comparisons honesty.** The robustness sweep across horizons × thresholds is itself a
  demonstration: on a *null* world, cherry-picking one horizon/threshold cut manufactures a
  spurious |*t*| > 2 — the exact multiple-testing trap the desk warns against (cf. Harvey, Liu &
  Zhu 2016, *"…and the Cross-Section of Expected Returns"*, on data-mined significance).

## Neighbours on this bench (the dedup map)

- **[Study 213 — Meme-Stocks](../../213-meme-stocks/)** — the meme-stock return phenomenon itself.
  Study 558 is narrower: the *FTD-spike-as-squeeze-predictor* microstructure signal specifically.
- **[Study 262 — Short-Interest](../../262-short-interest/)** — short interest / days-to-cover as a
  predictor. FTD is a *settlement-fail* metric, distinct from short interest (fails can occur
  without heavy short interest and vice versa).
- **[Study 308 — Cocoa-Squeeze](../../308-cocoa-squeeze/)** — a commodity squeeze study; 558 is an
  equity settlement-fail study.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a **real** tape for ``REAL``; synthetic-only caps at ``WEAK``/``NONE``), the explicit
  data-availability caveat, one execution lag, and costs one-way × NAV with shorts paying borrow.
