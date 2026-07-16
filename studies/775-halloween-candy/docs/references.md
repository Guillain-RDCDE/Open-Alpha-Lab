# References & literature map — Study 775 (Halloween-Candy)

## The claim under test

- **The folklore.** "Buy Hershey into Halloween — it's the candy giant's biggest selling
  season, so the stock front-runs the trick-or-treat haul." A perennial retail /
  financial-media seasonal trade: because Halloween is the single largest confectionery
  occasion of the US year (industry trade groups routinely put Halloween candy sales at the
  top of the seasonal calendar), Hershey (`HSY`) is supposed to *rally into* Oct 31 and,
  in the folklore's second half, *fade after* once the season's news is out.
- **Why it's a clean calendar test.** Halloween is a **fixed public holiday — always
  October 31** — so a "buy K sessions before, sell on the day" rule is calendar-known
  decades ahead and zero-look-ahead by construction. The dates are hardcoded in
  [`data.py`](../halloween_candy/data.py); the only anchoring choice is to use the last
  trading close on/before Oct 31 when the 31st falls on a weekend/holiday.
- **The efficient-markets prior.** A seasonal every analyst can put in their calendar, tied
  to sales that show up on a predictable schedule, is exactly what a semi-strong-efficient
  market should already price into the pre-season quarters — see Fama (1970, *Efficient
  Capital Markets*, JF). The desk's prior is that any "rally into" is arbitraged away.

## What the literature actually says about calendar seasonality

- **Seasonality & the "sell in May / Halloween indicator"** — Bouman & Jacobsen (2002,
  *American Economic Review*, "The Halloween Indicator, 'Sell in May and Go Away'") document
  a *market-wide* Nov→Apr vs May→Oct return gap. Note this is the **opposite** calendar
  anchor from a single-name run-up *into* Oct 31, and it is about the aggregate market, not
  Hershey — but it is the canonical academic "Halloween" seasonality reference, and it
  cautions that the Oct-31 boundary is a seam where market seasonality itself changes gear.
- **Firm-level seasonality in returns** — Heston & Sadka (2008, *JFE*, "Seasonality in the
  cross-section of stock returns") show stocks have persistent same-calendar-month return
  patterns. This is the strongest academic reason a *specific* seasonal like a Halloween
  run-up in a candy name could exist — and the reason it deserves a real test rather than a
  dismissal.
- **Sell-the-news / anticipation effects** — the idea that a known catalyst (here, a
  predictable selling season) is bid up beforehand and sold once realised is old market lore
  with a thin formal record; the closest academic cousins are post-earnings-announcement
  drift (Ball & Brown, 1968; Bernard & Thomas, 1989) and the scheduled-announcement premium
  (Savor & Wilson, 2016, *JFQA*). None of it says a *retail selling season* pays.
- **Attention & investor-catalyst effects** — Barber & Odean (2008, *RFS*) on
  attention-driven buying; Da, Engelberg & Gao (2011, *JF*) on search-based attention.
  These motivate *why* retail might crowd into a high-visibility seasonal like Halloween in
  a household-name candy stock — but attention is not, by itself, a tradable edge.

## Data & method

- **Real tape:** `HSY` and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. HSY's sub-1 beta
  to SPY is why we measure the *abnormal* return `HSY − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  Halloween years (the correct unit — not a daily panel); Wilson hit-rate interval; a
  20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net
  leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-Halloween run-up (and optional post-holiday fade) — the detector must recover a
  planted bump and stay quiet on the null. See [`strategy.py`](../halloween_candy/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Bouman, S. &
Jacobsen, B. (2002). The Halloween Indicator. **American Economic Review**. · Heston, S. &
Sadka, R. (2008). Seasonality in the cross-section of stock returns. **JFE**. · Ball, R. &
Brown, P. (1968). **JAR**. · Bernard, V. & Thomas, J. (1989). **JAR**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. &
Wilson, M. (2016). **JFQA**.*
