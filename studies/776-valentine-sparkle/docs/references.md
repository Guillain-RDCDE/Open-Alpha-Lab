# References & literature map — Study 776 (Valentine-Sparkle)

## The claim under test

- **The folklore.** "Buy the jeweller into Valentine's Day — the market front-runs the
  gifting quarter, then sells the news once the holiday is past." Signet Jewelers (Kay,
  Zales, Jared) is the dominant US mall jeweller, and Valentine's Day is the year's first
  big diamond/engagement-gift peak (after the December holidays), so retail and financial
  media periodically float the seasonal "sparkle rally" idea.
- **Why it's a clean calendar test.** Valentine's Day is **fixed at February 14** by the
  Gregorian calendar and known years ahead — a "buy K sessions before, sell on/after the
  14th" rule is calendar-known and zero-look-ahead by construction. The dates are hardcoded
  in [`data.py`](../valentine_sparkle/data.py); when the 14th lands on a weekend the event
  logic anchors on the last trading close on/before it.
- **The efficient-markets prior.** A gifting-season catalyst that recurs on the *same
  calendar date every year* is exactly what a semi-strong-efficient market should already
  price — a decades-old, universally-known seasonal is the last place to expect free
  abnormal return. See Fama (1970, *Efficient Capital Markets*, JF).

## What the literature actually says about calendar seasonals

- **Calendar anomalies and their fragility** — Lakonishok & Smidt (1988, *RFS*, "Are
  Seasonal Anomalies Real?") and Sullivan, Timmermann & White (2001, *JEconometrics*,
  "Dangers of Data Mining: the case of calendar effects") show that most calendar seasonals
  shrink or vanish out-of-sample and once you correct for the multiple-testing search over
  many candidate dates. A single-name Valentine's window is a textbook such candidate.
- **Retail-sales seasonality vs stock returns** — the gap between a firm's *sales* being
  seasonal (Valentine's, Mother's Day, Christmas are genuine jewellery-demand peaks) and its
  *stock* being predictable is the whole point: the sales calendar is public, so the returns
  should not be forecastable from it. Related: Kamstra, Kramer & Levi (2003, *AER*) on
  seasonal-affective / mood seasonality in returns — a reminder that plausible seasonal
  *stories* rarely survive as tradable return seasonals.
- **Anticipation / "buy the rumour, sell the news"** — the idea that a known catalyst is bid
  up beforehand and sold once realised has a thin formal record; the closest academic cousins
  are the pre-announcement drift and scheduled-announcement premium literatures (e.g. Savor &
  Wilson, 2016, *JFQA*, on scheduled macro announcements). None of it says a *retail holiday*
  pays.
- **Attention & investor-catalyst effects** — Barber & Odean (2008, *RFS*) on
  attention-driven buying and Da, Engelberg & Gao (2011, *JF*) on search-based attention
  motivate *why* retail might crowd a high-profile consumer holiday — but attention is not,
  by itself, a tradable edge.

## Data & method

- **Real tape:** `SIG` (Signet Jewelers, NYSE, relisted September 2008) and `SPY` daily
  adjusted (total-return) closes via [yfinance](https://github.com/ranaroussi/yfinance), one
  combined panel. SIG's above-1 beta to SPY is why we measure the *abnormal* return
  `SIG − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  Valentine's years (the correct unit — not a daily panel); Wilson hit-rate interval; a
  20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net
  leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-Valentine's run-up (and optional post-holiday fade) — the detector must recover a
  planted bump and stay quiet on the null. See [`strategy.py`](../valentine_sparkle/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Lakonishok, J. &
Smidt, S. (1988). Are Seasonal Anomalies Real? **RFS**. · Sullivan, R., Timmermann, A. &
White, H. (2001). Dangers of Data Mining. **Journal of Econometrics**. · Kamstra, M.,
Kramer, L. & Levi, M. (2003). **American Economic Review**. · Barber, B. & Odean, T.
(2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. & Wilson, M.
(2016). **JFQA**.*
