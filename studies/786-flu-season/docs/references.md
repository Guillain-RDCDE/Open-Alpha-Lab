# References & literature map — Study 786 (Flu-Season)

## The claim under test

- **The folklore.** "Buy the drugstores into flu season — CVS and the pharmacy chains get
  bid up ahead of their peak revenue window (flu shots, cold-and-flu OTC, cough/cold
  scripts) as the market front-runs the autumn/winter cold season." A perennial
  retail/financial-media seasonal trade: because the U.S. flu season is a recurring, known
  calendar event, the pharmacy tape is supposed to *rally into* it.
- **Why it's a clean calendar test.** The U.S. influenza surveillance season is defined by
  the CDC/WHO as **MMWR epidemiological weeks 40 to 20** — it *begins in early October*
  every year, a fixed calendar convention known years in advance. So a "buy K sessions
  before the October-1 season start, hold into it" rule is calendar-known and
  zero-look-ahead by construction. The anchor is hardcoded in
  [`data.py`](../flu_season/data.py) with no year-to-year slippage (unlike a company event
  date). Source: CDC "The Flu Season" / FluView surveillance definition.
- **The efficient-markets prior.** A seasonal catalyst everyone can put in their calendar is
  exactly what a semi-strong-efficient market should already price. Flu revenue is a small,
  well-telegraphed slice of CVS's diversified P&L (retail pharmacy + Caremark PBM + Aetna),
  so the desk's prior is that any "rally into flu season" is arbitraged away — see Fama
  (1970, *Efficient Capital Markets*, JF).

## What the literature actually says about seasonality & scheduled events

- **Calendar / seasonal anomalies** — the "Halloween indicator" / sell-in-May effect
  (Bouman & Jacobsen, 2002, *American Economic Review*) is the canonical documented
  calendar seasonal; note that October-through-winter overlaps with the strong-half-of-year
  seasonal, a confound our SPY benchmark is meant to strip out (we measure CVS *minus* SPY).
- **Industry / weather seasonality in fundamentals vs prices** — the distinction between a
  predictable *earnings* seasonal and a predictable *return* seasonal is central: if flu
  revenue is fully anticipated it should show in guidance, not in an abnormal price drift.
  Related: Chang, Hartzmark, Solomon & Soltes (2017, *RFS*) on seasonalities in returns and
  the difficulty of separating them from risk.
- **Attention & catalyst effects** — Barber & Odean (2008, *RFS*) on attention-driven
  buying and Da, Engelberg & Gao (2011, *JF*) on search-based attention motivate *why*
  retail might crowd a headline seasonal like "flu season," and therefore why any effect is
  as likely to be a sentiment blip as a fundamental edge — attention is not, by itself, a
  tradable signal.
- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968); Bernard & Thomas
  (1989, 1990). A flu season is not an earnings print, but the folklore borrows PEAD's
  intuition that prices drift around scheduled information; our test asks whether any such
  drift exists around the *season start* specifically, and finds essentially none.

## Data & method

- **Real tape:** `CVS` (CVS Health Corp.) and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. CVS's below-market
  beta (~0.6-0.8) is why we measure the *abnormal* return `CVS − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  flu-season years (the correct unit — not a daily panel); Wilson hit-rate interval; a
  20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net
  leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-season run-up (and optional in-season give-back) — the detector must recover a planted
  bump and stay quiet on the null. See [`strategy.py`](../flu_season/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Bouman, S. &
Jacobsen, B. (2002). The Halloween Indicator. **American Economic Review**. · Chang, T.,
Hartzmark, S., Solomon, D. & Soltes, E. (2017). **Review of Financial Studies**. · Ball, R.
& Brown, P. (1968). **JAR**. · Bernard, V. & Thomas, J. (1989, 1990). **JAR / JAE**. ·
Barber, B. & Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**.*
