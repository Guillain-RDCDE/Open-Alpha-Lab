"""Study 606 — OPEC Announcement Effect.

The folklore: **OPEC meeting days move oil** — volatility doubles on decision day and the
post-decision drift is tradable. We hardcode the full calendar of OPEC Conference and
OPEC+ (ONOMM) ministerial decision dates 2000-2026 (107 meetings, compiled from the OPEC
press-release archive), map each to its first tradable session on the WTI (CL=F), Brent
(BZ=F) and USO tapes, and test three things:

  1. **Volatility.** |return| and intraday range on decision days vs an event-free
     baseline — Welch t on |r|, Brown-Forsythe spread test, variance ratio, a
     random-calendar placebo and a bootstrap CI on the "vol multiple".
  2. **Signed drift.** Cumulative close-to-close drift day 0..+5 after decisions,
     one-sample t per event plus a Newey-West dummy regression on the daily tape.
  3. **Surprise continuation.** Does the day-0 sign continue over day +1..+5 (the
     "trade the announcement" folklore), and does it survive futures costs?

Engine:
  * ``data``     — the hardcoded meeting table (source-commented), cache-first yfinance
                   OHLC loaders, and a deterministic synthetic world with a *planted*
                   meeting-day vol multiple + drift (the machinery control).
  * ``strategy`` — event-day mapping, the vol/drift/continuation statistics with
                   HAC/Welch inference, the placebo, and the cost ledger.
"""
