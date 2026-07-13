# References & literature map — Study 728 ("people drink more beer in recessions")

## The claim under test

- **The folklore.** A recurring bit of investing/lifestyle wisdom that **alcohol — beer in
  particular — is recession-proof**: when incomes fall, people cut vacations and cars but
  not their cheap, habitual pint, so a brewer is a *defensive, counter-cyclical* stock that
  holds up (or beats the market) in a downturn. Popularised as the **"lipstick / beer /
  small-indulgence index"** idea and the broader **"sin stocks are defensive"** meme. The
  testable version: (H₁) a beer stock has a **low downside beta** (β⁻ < β⁺ < 1); (H₂) it
  **out-returns SPY during recessions**; (H₃) that survives real-time implementation.
- **Where the belief comes from.** Ethanol/alcohol demand is often cited as **income-
  inelastic**; U.S. brewers marketed "recession resistance" during 2008–09. But the
  empirical record is more "trading *down* to cheaper brands" than "drinking *more*":
  - Distilled Spirits Council / Nielsen and academic work on **alcohol demand elasticity**
    find beer demand is only mildly inelastic and shifts toward value brands in downturns
    (a *margin* headwind for brewers, not a tailwind).
  - Kerr, Greenfield et al., studies of **alcohol consumption over the business cycle**,
    generally find per-capita consumption is **pro-cyclical or flat**, not counter-cyclical
    — i.e. people tend to drink *less*, not more, when unemployment rises.

## The recession dating (the event windows)

- **NBER Business Cycle Dating Committee — US business-cycle peaks and troughs.** The
  authoritative recession chronology used here. The three contractions inside the sample:
  **2001-03 → 2001-11** (dot-com), **2007-12 → 2009-06** (Global Financial Crisis),
  **2020-02 → 2020-04** (COVID).
  https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
- **The announcement lag (the tradability kill).** The committee dates cycles **ex-post**:
  it announced the Dec-2007 peak in **Dec-2008** (~12 months late) and dated the Apr-2020
  trough in **Jul-2021** (~15 months late). https://www.nber.org/news/business-cycle-dating-committee-announcements
  A recession-conditional strategy is therefore a **look-ahead** unless it uses a real-time
  recession nowcast (Sahm rule, term spread, initial claims).

## The tradable names (what a public investor can actually buy)

- **Molson Coors Beverage Co. (`TAP`, NYSE).** A large-cap, mass-market brewer (Coors,
  Miller, Blue Moon) — the closest listed thing to a "consumer-staple beer" stock; low
  market beta.
- **The Boston Beer Company (`SAM`, NYSE).** Samuel Adams / Truly / Twisted Tea — a craft
  brewer with a growth-stock profile; listed 1995. A useful contrast: same industry, very
  different risk character.
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the "beats the market" claim invokes.
- *(Beat 7 extensions:)* `BUD` (AB InBev), `STZ` (Constellation), `HEINY` (Heineken),
  `DEO` (Diageo), and the sin-stock cousins `MO` / `PM`.

## Why "defensive" ≠ "counter-cyclical" — the finance

- **Downside beta as the defensiveness metric.** Ang, Chen & Xing (2006), *Downside Risk*
  (*Review of Financial Studies*): conditional (down-market) beta is the right lens for
  "does it fall less?" — and stocks earn a premium for *high* downside beta, so a genuinely
  low-β⁻ name should, if anything, earn *less*. We split at the market's sign (the standard
  bull/bear-beta convention) as in Bawa & Lindenberg (1977).
- **Low beta is not alpha.** A stock that swings less than the market (β < 1) mechanically
  loses less in crashes without any "special" property — that is beta, priced and free to
  replicate with cash + index. The CAPM alpha (Jensen, 1968), estimated with **Newey-West
  (1987)** HAC standard errors, is the test for excess *beyond* that beta — here
  insignificant.
- **Sin stocks & defensive sectors.** Hong & Kacperczyk (2009), *The Price of Sin*, and
  Fabozzi et al. (2008) document a modest long-run sin-stock premium — but it is a
  *valuation/neglect* story, **not** evidence of counter-cyclical recession outperformance,
  and it does not appear in these two single names' recession windows.
- **Small samples and event studies.** With only three recessions (31 months), a
  recession-window test is low-power and hostage to one idiosyncratic name-quarter — the
  multiple-comparisons / selection warning of White (2000) and the desk's Reality-Check
  discipline applies: an apparent "edge" driven by two rallies is not a law.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../beer_recession/strategy.py)).
- **Downside defensiveness.** Bull/bear conditional beta
  ([`strategy.bull_bear_beta`](../beer_recession/strategy.py)); defensive ⟺ β⁻ < β⁺ < 1.
- **Robust inference.** A **Newey-West (HAC)** *t* of the CAPM alpha vs SPY
  ([`strategy.newey_west_alpha_t`](../beer_recession/strategy.py)) and a paired recession-
  window excess *t* ([`strategy.recession_excess_t`](../beer_recession/strategy.py)).
  `REAL` would require `|t| ≥ 2` **in the beer's favour** — no leg clears it.
- **Deterministic synthetic control.** A fixed-seed planted-asymmetric-beta generator
  ([`data.synthetic_defensive`](../beer_recession/data.py)) proving the engine recovers a
  real defensive tilt — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `TAP`, `SAM`, `SPY`, cached under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).
- **NBER** recession dates as above (hardcoded, cited; a dated set of facts, not a feed).

## Related desk studies

- **[Study 358 — Watches are an asset class?](../../358-watch-index/)** — the same
  "collectible/passion-asset beats stocks" teardown shape (labelled proxy + tradable legs
  vs SPY), and the same survivorship-narrated-as-system signature.
- **[Study 550 — Box-office momentum](../../550-box-office-momentum/)** — the consumer-
  oddity-on-a-single-stock family this study belongs to.
