# References & literature map — Study 710 (Olympic-Host-Effect)

## The claim under test

- **The folklore.** "Hosting the Summer Olympics is good for the host's stock market" —
  national pride, a wave of construction spending, a tourism bump and years of positive
  media attention should, the story goes, show up as an abnormal return in the host's
  equity market around the Games. Every host bid ("Sydney will reap an Olympic dividend",
  "Tokyo 2020 will be an economic springboard") is sold partly on this premise.
- **The academic anchor is mixed, and mostly bearish on the "winner's curse".** Baade &
  Matheson (2016, *Going for the gold: the economics of the Olympics*, JEP) survey decades
  of host-city cost-benefit studies and find promised economic benefits routinely fail to
  materialize — the Games are, on the whole, a "winner's curse" for the host city's public
  finances (systematic cost overruns: Flyvbjerg, Stewart & Budzier 2016, *The Oxford
  Olympics Study*). Berman, Brooks & Davidson (2000, *The Olympic effect*, working paper)
  is the closer financial-markets anchor: they find a modest POSITIVE abnormal stock
  return around the announcement of the host award — an event distinct from the Games
  themselves. This study tests neither the award-announcement pop nor the fiscal
  cost-benefit ledger; it tests the plain-vanilla "market rallies around the Games" claim
  in the window investors actually talk about.
- **The stronger sibling result — the World Cup — sets the bar.** [235-world-cup-effect](../../235-world-cup-effect/)
  finds a WEAK, confounded signal for a *different* mega-event, at n = 19 tournaments and a
  much shorter (~5-week) window. This study runs the analogous test for the Olympics'
  HOST country specifically, at a much smaller n (6, one host-country ETF per edition) and
  a much longer window (8 months) — a harder sample-size bar from the outset, named openly
  rather than glossed over.

## What we measure, and the honesty rails

- **Abnormal return** = host-ETF total return over [-6mo..+2mo] around the opening/closing
  ceremony MINUS the ^GSPC price return over the identical calendar window (nearest
  trading day). One documented execution convention: enter at the window-start trading
  day's close, exit at the window-end trading day's close — host cities are awarded by IOC
  vote 7-9 years ahead of the Games, so this is a zero-look-ahead, calendar-known event
  exactly like a scheduled announcement.
- **n = 6, not 7 — said out loud.** Athens 2004 sits in the hardcoded host table with no
  ticker: no single-country Greece ETF existed in 2004 (Global X's GREK launched
  2011-12-08, seven years after the Games closed). Rather than force a post-hoc proxy onto
  a market with no contemporaneous listed vehicle, the real-tape panel runs on the six
  hosts that DO have one, and every table says n = 6.
- **Every named confounder is exactly that — named, not laundered.** Beijing 2008 sits
  inside the Global Financial Crisis (Lehman failed 2008-09-15, squarely inside this
  window's post-Games leg). Rio 2016 coincides with Brazil's rebound off the 2016
  commodity-bust trough (Bovespa bottomed January 2016) — a macro story with nothing to do
  with hosting. Both confounders are named on the Signal axis, not buried in a footnote.
- **Tiny-n inference discipline.** A single one-sample t on six numbers is exactly the kind
  of statistic one fat-tailed observation can flip — so alongside the t we report the
  median, a Wilcoxon signed-rank test, a percentile bootstrap CI, and a random-window
  placebo (same tickers, same window LENGTH, random calendar anchor) so no single method
  carries the whole verdict.
- **The world benchmark is ^GSPC, a named substitute for URTH/ACWI.** iShares' MSCI World
  ETF (URTH, inception 2012-01-04) and MSCI ACWI ETF (ACWI, inception 2008-03-26) both
  postdate part of the 2000->2024 sample this claim spans (ACWI's inception even sits
  inside the Beijing 2008 pre-Games leg). ^GSPC runs continuously since 1927 with no
  survivorship, at the cost of being a US, price-only (no-dividend) proxy for "the world" —
  named wherever it appears, and every table labels host **total return** vs benchmark
  **price-only** return.

## Data sources

- **Host-country ETF adjusted (total-return) closes**: EWA, FXI, EWU, EWZ, EWJ, EWQ — and
  **^GSPC** (S&P 500, price-only) — all from yfinance (no key), cached under `_cache/`
  (`ohe_<ticker>.csv`), 1998-01-01 -> 2026-06-30.
- **Summer Olympics host calendar 2000 -> 2024**, hardcoded in
  [`data.py`](../olympic_host_effect/data.py) — opening/closing-ceremony dates from the
  IOC's own results archive: https://olympics.com/en/olympic-games (per-edition results
  pages). Host-city award dates (all 7-9 years ahead of the Games) confirm the zero-
  look-ahead framing but are not themselves used in the window construction.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [234-olympic-year](../../234-olympic-year/) — does the **US** market (^GSPC) do better in
  **any** Summer-Olympic **year**, host or not? A calendar-year effect on one market. This
  study: the **specific host country's** market in an **8-month event window**, not a
  calendar-year effect and not restricted to the US.
- [235-world-cup-effect](../../235-world-cup-effect/) — the analogous **World Cup** test on
  the **global** (S&P 500) market during the tournament window — a different mega-event,
  a different (non-host-specific) market, a shorter window. This study is host-country-
  specific, for the Olympics.
- [708-eurovision-effect](../../708-eurovision-effect/) — the same "does hosting a mega
  cultural/sporting **event** lift the host market" question, asked of Eurovision, a far
  smaller and shorter event. A sibling test of the same mechanism on a different event
  class; results are independent (different calendars, different confounders).
- [313-geopolitical-shock](../../313-geopolitical-shock/) — event-study machinery (constant-
  mean abnormal return, placebo, bootstrap) applied to **shocks** (wars, attacks), the
  opposite valence and a different trigger class entirely. Shared method, unrelated claim.

None of the siblings test what this study tests: the **specific host country's** equity
market, in the **[-6mo..+2mo] Games window**, against a **world benchmark** — the "national
pride rally" claim on its own terms.
