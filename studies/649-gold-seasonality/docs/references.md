# References & literature map — Study 649 (Gold Seasonality)

## The claim under test

- **The folklore.** Gold desks and financial-media roundups repeat a two-part calendar every
  autumn: **September is "gold's best month"** — driven by Indian **wedding-season** and
  pre-**Diwali** physical jewellery demand (Diwali itself typically falls in October/November,
  but the restocking and hedging that precedes it is dated to September) plus Northern-hemisphere
  jewellers restocking ahead of the year-end holiday season — while the **Northern-hemisphere
  summer** (roughly May–August, between the spring Akshaya Tritiya buying window and the autumn
  wedding season) is a quiet "lull" in physical offtake. LBMA/WGC market commentary and countless
  seasonal-return roundups (Seasonax, EquityClock, and similar commercial seasonality-chart
  vendors) repeat versions of this table every year.
- **The academic anchor.** The evidence for *any* robust month-of-year effect in broad asset
  seasonality is thin and contested — the "Halloween effect" / "Sell in May" literature (Bouman &
  Jacobsen 2002, *The Halloween indicator, "Sell in May and go away": Another puzzle*, AER) finds
  a robust equity effect but explicitly does **not** extend it to commodities; gold-specific
  seasonality studies (e.g. Lucey & Tully 2006, *Seasonality, risk and return in daily COMEX gold
  and silver data 1982–2002*, Applied Financial Economics) report **weak, inconsistent** monthly
  patterns that do not survive out-of-sample or multiple-testing correction. The **India physical
  demand** mechanism itself is real and well documented (World Gold Council *Gold Demand Trends*,
  quarterly) — Indian jewellery/investment demand genuinely clusters around Akshaya Tritiya
  (spring) and the wedding-and-festival season (autumn) — but a real *physical-market* seasonal
  does not automatically imply a tradable *price* seasonal on a US-listed, dollar-denominated ETF
  whose price is set by the much larger global futures/ETF-flow market.
- **The adjacent (distinct) result.** [289-diwali-muhurat](../289-diwali-muhurat/) tests the
  *equity* Muhurat-session omen in *India* — the same festival, a completely different asset and
  market. Neither that study nor this one shares a claim with the other; see the dedup map below.

## What we measure, and the honesty rails

- **Month-of-year mean returns** — 12 calendar-month cells on GLD's own monthly log return,
  one-sample naive and **Newey-West (1987)** HAC *t*-stats. A **Bonferroni** correction
  (α = 0.05/12) is the honesty rail against reporting one lucky month out of 12 draws as "the"
  seasonal — exactly the discipline used by sibling seasonality studies
  [648-grain-seasonality](../648-grain-seasonality/) and
  [307-coffee-seasonality](../307-coffee-seasonality/).
- **September vs the rest, summer vs the rest** — Welch *t* of the pooled claimed-strong /
  claimed-weak group against every other month pooled, plus a **circular block-bootstrap CI**
  (5,000 draws, 12-month blocks, respecting the annual seasonal structure) on the September
  spread.
- **Era contrast (2013-04-01)** — the 2013-04-12/15 gold crash (spot gold fell ~9% on 2013-04-15
  alone, its worst single day since 1980, and ~13% over two sessions) is the externally-dated,
  justified break between the 2001–2012 bull "supercycle" and gold's subsequent decade-long
  range: pre/post September means, within-era Welch *t*'s, and a Welch *t* of the *difference* —
  never eyeballed.
- **Calendar-known execution.** The "own gold only in September" timer sets its position from the
  fixed calendar alone — September is the same slot every year, known well in advance — so the
  study's single documented execution convention needs **no signal-to-trade lag** (unlike a
  data-driven signal, a calendar rule is knowable at the start of the year; the monthly return
  itself already spans the August close → September close).
- **Costs charged one-way × NAV per leg** (5/10 bps typical), 2 legs on the single active month
  per year, exactly the convention used by [637-fomc-vol-crush](../637-fomc-vol-crush/) and
  [648-grain-seasonality](../648-grain-seasonality/).
- **Sharpe races are excess-of-cash vs excess-of-cash.** Both the timer and buy-and-hold are
  measured against the ^IRX 13-week T-bill yield, so the timer isn't flattered simply for being
  out of the market (and therefore lower-vol) 11 months a year.

## Data sources

- **GLD** daily adjusted closes (SPDR Gold Shares, inception 2004-11-18) and **^IRX** daily
  closes (13-week T-bill discount yield) — yfinance (no key), cached under `_cache/`
  (`gsn_gld.csv`, `gsn_irx.csv`), 2004-11-18 → 2026-06-30. GLD holds allocated physical bullion —
  no futures roll, no dividend to adjust for — so its daily close is a direct, tradable spot-gold
  proxy.
- **World Gold Council**, *Gold Demand Trends* (quarterly): https://www.gold.org/goldhub/data/gold-demand-by-country
  — the primary source for the India-demand mechanism the claim invokes.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [289-diwali-muhurat](../289-diwali-muhurat/) — the Diwali Muhurat-session **equity** omen,
  **India**, a single one-hour session. Different asset class, different market, a one-day event
  rather than a whole-month calendar. This study is gold's own *monthly* seasonality on a US ETF.
- [69-safe-haven](../69-safe-haven/) — whether gold hedges inflation or crashes (a
  *cross-sectional macro-regime* question). No calendar, no month-of-year axis.
- [580-gold-lease-rate](../580-gold-lease-rate/) — the borrow-cost **lead-lag** (a microstructure
  carry signal, and synthetic-only for want of a real tape). Not a calendar effect.
- [640-gold-overnight](../640-gold-overnight/) — the **intraday clock** (overnight vs intraday
  session return), a daily/session-level pattern, not a month-of-year one.
- [305-gold-oil-ratio](../305-gold-oil-ratio/) — a cross-asset **ratio-timing** signal (regime
  risk-on/risk-off), no calendar axis at all.
- [113-gold-silver-ratio](../113-gold-silver-ratio/) — a mean-reversion **pairs trade** between
  two metals, no calendar axis.

None of the siblings test the **month-of-year (September strength / summer lull) calendar** on
gold — this study's own axis.
