# References & literature map — Study 735 (Ryder-Cup-Effect)

## The claim under test

- **The folklore.** The Ryder Cup — the biennial team match between **Team USA** and
  **Team Europe** — is one of the few genuinely *continental* sporting rivalries, played
  for pride rather than money, with enormous television reach on both sides of the
  Atlantic. The sports-sentiment folklore says a national (here *continental*) mood shock
  should move markets: the *losing* continent's stock market should underperform the
  winner's the Monday after the Sunday result. It is a two-sided, cross-market version of
  the one-directional football effect below.
- **The academic anchor (a different sport, a real, one-sided effect).** Edmans, García &
  Norli (2007, *Sports Sentiment and Stock Returns*, **Journal of Finance** 62(4)) find a
  robust next-day market decline after a country is **eliminated** from soccer's World Cup
  (and, more weakly, other tournaments) — a genuine *loss*-driven mood-to-market channel,
  economically and statistically significant, with **no symmetric win effect** (winning
  does *not* lift the market). Ashton, Gerrard & Hudson (2003, *Economic impact of
  national sporting success on the FTSE 100*, **Applied Economics Letters** 10) find a
  same-day England-football-result effect on the FTSE. Kaplanski & Levy (2010, *Sentiment
  and Stock Prices: The Case of Aviation Disasters*, **JFQA** 45) is the same
  mood-to-market mechanism from a non-sport shock. The Ryder Cup borrows the *loss*
  mechanism but applies it to a contest whose fanbase, betting volume and emotional
  stakes — while real — are a fraction of a football World Cup's, and in which "the loser"
  is an entire *continent's* team, not a single nation.
- **What nobody has published.** There is no peer-reviewed study of a Ryder-Cup stock
  effect that we are aware of — this is a plausible-sounding extrapolation of the Edmans
  mechanism, not a tested academic claim, and Edmans et al. explicitly find **no win-side
  effect**, which already lowers the prior for a symmetric "loser lags, winner leads"
  pattern. The desk starts this one with a low prior.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Wikipedia's "Ryder Cup" and
  per-edition pages: every edition of the modern USA-vs-Europe era (1979→2025), with the
  concluding date, the winning side and the losing side. **1989 was a 14-14 tie** (Europe
  retained the Cup) — there is no losing side, so it is a named, excluded row. 2001 (9/11)
  and 2020 (COVID) were postponed to 2002 and 2021, and the calendar carries the year the
  match was actually *played*. **2010** (Celtic Manor) was rain-delayed and concluded on
  the **Monday** — the engine snaps the first post-result close to that Monday, which
  already knew the outcome, so the lag stays zero-look-ahead without special-casing.
- **The paired instrument.** Team USA → `SPY` (S&P 500 ETF), Team Europe → `VGK` (Vanguard
  FTSE Europe). The primary statistic is the **loser-minus-winner** return, so the common
  global weekend move cancels and what remains is the cross-Atlantic *relative* move the
  folklore is actually about.
- **Why VGK, not FEZ.** FEZ (Euro Stoxx 50) is Eurozone-only. The Ryder Cup's European
  team is drawn from all of Europe — the United Kingdom, Switzerland, Sweden, Norway,
  Denmark are core sources of European players and are **not** in the euro. VGK (Vanguard
  FTSE Europe) spans both euro and non-euro Europe and is the only fair "Team Europe"
  proxy. VGK's inception (2005-03-10) is the study's hard floor: only the **2006→2025**
  editions are testable (10 events, 7 USA-losses to 3 Europe-losses) — a real
  power constraint, disclosed rather than papered over with a pre-2005 index proxy.
- **One documented execution lag.** The result lands Sunday (non-trading). day(-1) = the
  last common close before the result (does not know the winner); day(0) = the first
  common close after (fully public). The **signal** measurement runs day(-1)→day(-1)+k
  (the full announcement reaction, including the un-tradable weekend jump); the
  **tradability** measurement enters at day(0)'s close instead (long winner / short loser)
  — zero look-ahead by construction, and the honest gap between "an effect exists" and
  "you could have banked it."
- **Inference unit.** Each Ryder Cup is one independent, non-overlapping biennial event —
  the correct test is a **one-sample t** of the paired spread across events, not a daily
  panel regression. A random-window placebo (drawing many non-Ryder-Cup k-session windows
  on the *same* pair, keeping each event's loser/winner assignment) checks whether the
  observed mean sits outside the pair's ordinary week-to-week noise.
- **The Edmans mechanism, tested directly.** A third-axis constant-mean (Brown & Warner
  1985) leg decomposition asks whether the *loser's own* market slumps (the actual Edmans
  channel) or whether any paired gap is just the winner's leg being weak — the difference
  between "sports sentiment moved this" and "two noisy legs in a soft calendar month."

## Data sources

- **Daily adjusted (total-return) closes** for `SPY` and `VGK` — yfinance (no key), cached
  under `_cache/`.
- **Ryder Cup winners/losers and concluding dates, 1979→2025** — hardcoded in
  [`data.py`](../ryder_cup_effect/data.py). Sources: Wikipedia, "Ryder Cup"
  (https://en.wikipedia.org/wiki/Ryder_Cup) and the per-edition "YYYY Ryder Cup" pages,
  cross-checked for the concluding Sunday (and the 2010 Monday finish).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — the same "national-pride bump"
  shape applied to a *cultural* contest, tested as a **per-country** abnormal-return panel
  vs VGK. This study is different on two counts: it is a **paired, two-market
  loser-minus-winner** spread (USA vs Europe), and it tests the *loss* side of the Edmans
  mechanism (the half that is actually real for football), not a *win/host* bump.
- [235-world-cup-effect](../../235-world-cup-effect/) — the real Edmans elimination effect
  for football World Cups on the S&P 500. The closest academic anchor to this study's
  mechanism, but a single-market, single-tournament window, not a cross-Atlantic pair.
- [707-plane-crash-effect](../../707-plane-crash-effect/) — the Kaplanski-Levy
  mood-to-market shock from a non-sport tragedy, an event study on SPY with a
  random-calendar placebo and a paired sector-extra-drop test. Same event-study /
  placebo / synthetic-control machinery; different trigger and a single market.
- [158-super-bowl](../../158-super-bowl/) / [709-world-series-effect](../../709-world-series-effect/)
  / [234-olympic-year](../../234-olympic-year/) — other sports-folklore calendar claims on
  single national indices. None of them test a **paired cross-continental loser-minus-winner
  spread keyed to a biennial team match** — that framing, and the "the loser's leg doesn't
  even slump; if anything the winner's does" finding, is this study's own contribution.
