# References & literature map — Study 656 (Dragon Portfolio)

## The claim under test

- **The paper.** Christopher Cole (Artemis Capital Management), *"The Allegory of the
  Hawk and Serpent"* (2020) — the "Dragon Portfolio". Cole builds a **100-year**
  backtest (1928-2020) of a 5-sleeve mix — global equities, fixed income, gold,
  commodity trend-following and long volatility — and argues it survives every
  historical regime (the 1930s deflation, the 1970s inflation, 2008, 2020) that breaks
  a plain 60/40 book, because each sleeve is chosen to *profit*, not merely diversify,
  in its regime. The publicly quoted headline allocation is **24% equities / 18% fixed
  income / 19% gold / 18% commodity trend / 21% long volatility**.
- **The mechanism, steelmanned.** Stocks and bonds both do well in "growth" regimes and
  both can fail together (2022) when inflation and rates rise at once — the classic
  60/40 blind spot. Gold and commodity trend are meant to profit from inflationary
  regimes; long volatility is meant to profit from the deflationary liquidity crises
  (1930s, 2008, March 2020) where *everything else* — including gold, briefly — sells
  off together. The "long vol" sleeve is the piece 60/40 (and even the Permanent
  Portfolio and All-Weather) structurally lacks.
- **The academic-adjacent anchor.** The convexity/crisis-alpha argument echoes
  Taleb & Spitznagel's tail-hedging literature (*The Black Swan*, 2007; Spitznagel,
  *Safe Haven*, 2021) and the broader "crisis alpha" managed-futures literature
  (Kaminski, *Trend Following with Managed Futures*, 2014). This desk's own
  [617-crash-insurance-cost](../../617-crash-insurance-cost/) already measures what a
  live tail-hedge product costs; this study asks the allocation-level question — does
  *bundling* a small long-vol sleeve into a broader mix pay for itself.

## What we measure, and the honesty rails

- **We cannot replicate Cole's own 100-year backtest.** Cole's paper uses academic
  replicated/synthetic indices (bond, gold, trend, vol proxies) reaching back to 1928 —
  those series are not on public tape. We proxy the five sleeves with **live, liquid
  ETFs** on yfinance (SPY / TLT / GLD / a 12-month trend overlay on DBC / VXX), which
  caps the testable window at whichever sleeve's data starts latest — and that is a
  **decade**, not a century. Any claim about "the 100-year secular cycle" is, on this
  data, **untestable by construction**; we say so and test only what the real tape
  can show: two recent regime shocks (2020 deflationary, 2022 inflationary).
- **The long-vol sleeve is a deliberately crude, decaying proxy, named as such.** VXX
  (iPath Series B S&P 500 VIX Short-Term Futures ETN) holds front-month VIX futures and
  bleeds on contango in most months; it is **not** what a real long-vol book (actively
  rolled and sized OTM SPX puts / variance swaps) would hold — it is simply the only
  continuously tradable long-vol instrument on public tape. **Named data quirk:** the
  VXX product dates to 2009-01-30, but Barclays halted and relaunched it as "Series B"
  in Jan-2018 and yfinance's own VXX tape starts **2018-01-25** — the real binding
  constraint on the whole 5-sleeve backtest, nearly a decade shorter than the product's
  nominal history.
- **The commodity-trend sleeve is also a simplification.** Cole's "trend" sleeve is a
  diversified systematic program across dozens of futures curves (rates, currencies,
  commodities, equities); we proxy it with a single 12-month time-series-momentum
  overlay on one commodity index (DBC). The side-check against DBMF (a real
  managed-futures ETF, live since 2019) shows only **+0.28 daily correlation** over
  their overlap — a materially weaker proxy than the ticker choice alone suggests,
  named explicitly in [results.md](results.md).
- **Two regime episodes, not a certified sample.** 2020 (COVID) and 2022 (the classic
  stocks-and-bonds both-down inflation year) are the cleanest available deflationary
  and inflationary shocks in the testable window. Both are reported narratively — no
  *t*-stat is claimed on n=2 episodes (house rule: no conditional claim without
  uncertainty applies to *aggregate* stats, which we do certify with HAC/bootstrap
  separately; single historical episodes are described as exactly that).
- **One execution lag, documented exactly.** The commodity-trend flag (DBC's trailing
  12-month return, evaluated at the prior month's close) decides the following month's
  position — the study's only forecast, calendar-known, zero look-ahead. Every other
  sleeve's weight is an exogenous constant reset monthly; no lag needed there.

## Data sources

- **SPY / TLT / GLD / DBC / VXX / SHY / DBMF** daily auto-adjusted (total-return)
  closes — yfinance (no key), cached under `_cache/` (`dragon_*.csv`), 2002-01-02 →
  2026-06-30 (per-ticker inception binds the earliest usable date; see
  [`dragon_portfolio/data.py`](../dragon_portfolio/data.py)).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- Cole, C. (2020). *The Allegory of the Hawk and Serpent.* Artemis Capital Management.
- Kaminski, K. (2014). *Trend Following with Managed Futures: The Search for Crisis
  Alpha.* Wiley.
- Spitznagel, M. (2021). *Safe Haven: Investing for Financial Storms.* Wiley.

## Related desk studies (the dedup map — what this study is NOT)

- [68-all-weather](../../68-all-weather/) — Ray Dalio's **risk-parity** (inverse-vol
  weighted) SPY/IEF/GLD/DBC blend. This study's "All-Weather-lite" benchmark is a
  *static equal-weight* cousin of the same four tickers, used only as a same-window
  comparison point — it is **not** 68's risk-parity engine, and doesn't include a
  vol or trend-timing sleeve. For the rigorous inverse-vol version, see 68.
- [144-permanent-portfolio](../../144-permanent-portfolio/) — Harry Browne's static
  25/25/25/25 SPY/TLT/GLD/SHY, annually rebalanced, no trend-timing and no vol sleeve
  at all. Dragon adds two sleeves (systematic trend, long vol) 144 doesn't have, and
  rebalances monthly rather than annually.
- [617-crash-insurance-cost](../../617-crash-insurance-cost/) — measures what a
  **live, standalone** tail-risk product (TAIL) costs a buy-and-hold investor. This
  study asks a related but distinct question: does folding a *small* long-vol sleeve
  into a broader multi-asset mix pay for itself at the allocation level — the VXX
  standalone diagnostic here (CAGR −40.7%/yr) is directionally consistent with 617's
  finding that crash insurance bleeds, using a different, cruder instrument (VXX vs
  TAIL) and a different question (allocation drag vs standalone cost).
- [655-ivy-portfolio](../../655-ivy-portfolio/) — Mebane Faber's 5-asset (equity /
  international equity / REITs / bonds / commodities) 10-month-SMA timing overlay
  applied to *every* sleeve. Dragon applies trend-timing to **only** the commodity
  sleeve (the other four are static-weight) and adds a long-volatility sleeve Ivy
  doesn't have — a different regime-survival thesis (crisis-alpha via convexity) vs
  Ivy's broad trend-following risk reduction.

None of the siblings test the specific claim this study does: does **bundling a
long-volatility sleeve** into a multi-asset mix make it survive both inflationary and
deflationary regimes where 60/40 fails — and can that sleeve be built cheaply off the
shelf.
