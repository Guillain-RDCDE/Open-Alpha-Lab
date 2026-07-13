# References & literature map — Study 769 ("Parks attendance/pricing as a DIS tell")

## The claim under test

- **The pitch.** A recurring alt-data / consumer-tell claim heard from retail investors and
  fan-finance commentary: **theme-park attendance and Disney's ticket-pricing power are a
  leading indicator for `DIS` the stock** — "the parks are packed and prices keep rising, so
  Disney must be a buy." The Parks, Experiences & Products segment is Disney's single largest
  profit centre, so the intuition that park demand *tells you something* is not crazy. The
  testable version: does a **strictly-lagged** parks-momentum signal *lead* DIS's forward
  return — specifically its return **in excess of the market** — net of costs?
- **Why the lag is the whole game.** The signal is only worth something if you can act on it
  *before* the market does. The industry attendance figures arrive with a large, real lag
  (below), and Disney reports Parks & Experiences segment revenue **quarterly** — so by the
  time the annual attendance print lands, the market has seen ~three quarters of the same story.

## The parks data (the "real tape" we proxy)

- **TEA/AECOM — *Theme Index & Museum Index*.** The industry-standard annual report of
  attendance at the world's top amusement/theme parks and park groups, produced by the
  **Themed Entertainment Association (TEA)** with **AECOM**. It is the source almost every
  "most-visited parks" headline traces back to. Published as an annual PDF, historically
  released in the **middle of the following year** (~May–July), and **not a free API** — hence
  our hardcoded, cited, *approximate* annual reconstruction and its encoded release lag.
  https://www.teaconnect.org/Resources/Theme-Index/index.cfm ·
  AECOM: https://aecom.com/theme-index/
- **Walt Disney Company 10-K / segment filings.** Disney reports **Parks, Experiences and
  Products** (formerly Parks & Resorts) segment revenue and operating income *quarterly* — the
  real-time, higher-frequency version of "how are the parks doing" that front-runs the annual
  Theme Index. https://thewaltdisneycompany.com/investor-relations/
- **Ticket-price history (pricing-power proxy).** Walt Disney World Magic Kingdom one-day
  peak base ticket prices are widely tracked in the trade press (e.g. AllEars, WDW Magazine,
  news coverage of each increase): base ~$79 in 2010 rising to ~$159 (date-based pricing from
  2018-19) and ~$199 peak by 2024. Increases are announced immediately (public the day of the
  change), so the pricing signal has essentially **no** lag — a useful contrast to attendance.

> **Transparency.** `disney_parks.data.load_attendance` and `load_ticket_price` are **small,
> hardcoded, approximate** annual series whose *path* matches the public TEA/AECOM Theme Index
> and reported ticket prices (steady 2010s growth, the 2020 COVID crater, the 2021-23
> recovery). They are **labelled proxies for the real tape, never the real tape**, and the
> study's verdict reflects that limitation.

## Why "an announced/late alt-data print leads the stock" is the wrong default — the finance

- **Semi-strong market efficiency.** Fama (1970), *Efficient Capital Markets: A Review of
  Theory and Empirical Work*, and Fama (1991). Public information — a quarterly segment report,
  a widely-covered annual attendance index — is impounded into prices quickly; a signal you
  learn *after* it is public should not predict abnormal (excess) returns.
- **Post-earnings / information-timing.** The relevant park information reaches the market
  through Disney's quarterly segment disclosures (Ball & Brown 1968; Bernard & Thomas 1989 on
  post-earnings-announcement drift). A once-a-year attendance print released six-plus months
  late is *stale* relative to that flow — the classic alt-data trap of a slow, low-frequency,
  already-priced series.
- **Segment ≠ whole company.** Parks are one segment; DIS's price is also driven by the Media
  & Entertainment / streaming (Disney+) business, content spend, buybacks, guidance and macro
  beta. A parks-only tell is a partial signal for a multi-segment stock — attenuation by
  construction.
- **Alt-data decay.** Once an alternative dataset is widely known and cheaply available, its
  edge decays toward zero (the general lesson of the alt-data literature; see e.g. the
  crowding/decay discussion in McLean & Pontiff 2016, *Does Academic Research Destroy Stock
  Return Predictability?*). A public, journalism-friendly index is the *least* edgy kind.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../disney_parks/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* of DIS vs SPY
  ([`strategy.annual_excess_t`](../disney_parks/strategy.py)); a **Newey-West (HAC)** *t* of
  the lead-lag slope with Bartlett weights for the overlapping forward windows
  ([`strategy.newey_west_slope_t`](../disney_parks/strategy.py), following Newey & West 1987);
  and a Welch *t* for the regime contrast ([`strategy.regime_split`](../disney_parks/strategy.py)).
  `REAL` would require a HAC/Welch *t* ≥ 2 **on DIS's excess return** — none clears it.
- **Cost realism (beat 6).** A rotation timing backtest with a one-month execution lag and a
  per-leg cost on each switch ([`strategy.timing_backtest`](../disney_parks/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed momentum-plus-planted-edge generator
  ([`data.synthetic`](../disney_parks/data.py)) proving the engine recovers a planted signal
  and stays null when none is planted — runs with no network.
- **Look-ahead discipline.** The Theme-Index release lag (July Y+1) is encoded in
  [`data.release_date`](../disney_parks/data.py); the strategy adds one further execution lag.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `DIS`, `SPY`, cached under `_cache/`.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- **Hardcoded parks series** as above (public TEA/AECOM Theme Index + reported ticket prices;
  approximate; labelled proxies).

## Related desk studies

- **[Study 358 — Watches are an asset class?](../../358-watch-index/)** — the same
  labelled-proxy discipline: a hardcoded, cited, approximate index tested honestly against SPY.
- **[Study 708 — Eurovision effect](../../708-eurovision-effect/)** — another cited-proxy
  alt-data curio aligned to equities with a strict no-look-ahead lag.
- **[Study 387 — Economic-Surprise-Index](../../387-economic-surprise-index/)** — the
  release-lagged macro-tell shape (a slow public signal that is already priced by the time you
  can act on it).
