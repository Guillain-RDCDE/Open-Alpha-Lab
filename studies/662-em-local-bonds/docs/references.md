# References & literature map — Study 662 (EM-Local-Bonds)

## The claim under test

- **The folklore.** "Local-currency EM bonds pay a fatter yield than USD EM debt because you're
  compensated for currency risk *and* sovereign-credit risk — the extra pickup is real carry,
  not a free lunch, but it's a *paid-for* one." The pitch typically points at the running-yield
  gap between the J.P. Morgan GBI-EM (local) and EMBI Global Diversified (USD hard-currency)
  indices — often several hundred basis points — and argues that gap compensates a diversified,
  patient holder.
- **The academic anchor.** Bekaert & Wang (2012, *Local currency bond markets*, Emerging Markets
  Review) document the structural drivers of local EM debt returns and its FX component;
  Burger, Warnock & Warnock (2015, *VoxEU / NBER*, "Bond market development in developing
  Asia" and related work) trace the post-2000 growth of local-currency EM sovereign issuance as
  a deliberate policy response to the "original sin" (currency mismatch) that drove hard-currency
  EM defaults in the 1980s–90s. The pitch is that local debt *removes* the mismatch for the
  sovereign — but that only shifts the currency risk onto the **holder**, which is exactly the
  axis this study measures.
- **The adjacent (distinct) result.** [612-em-debt-carry](../612-em-debt-carry/) tests the
  **USD-denominated** side of the same asset class — EMB's coupon pickup over IEF — and finds
  the *promised* coupon real (HAC *t* ≈ 20) but the *collected* total-return spread
  uncertifiable (*t* = 0.99) once the credit-beta/crisis left tail is priced in. That study never
  touches currency; this one is entirely about the currency leg 612 sidesteps by construction.

## What we measure, and the honesty rails

- **Local basket = average of EBND and LEMB**, two ETFs tracking *different* index families
  (Bloomberg EM Local Currency Government Universal vs J.P. Morgan GBI-EM Global Diversified) —
  a cross-provider robustness check on the same asset class, not double-counting one benchmark.
- **The headline is a paired monthly excess-of-cash spread** (Local minus EMB), tested with a
  paired one-sample *t*, a Newey-West (1987) HAC *t* at three lag choices (3/6/12 months — bond
  total returns are serially correlated at typical rebalance/settlement horizons), and a
  **circular block bootstrap** (Politis-Romano-style resampling, 6-month blocks) 95% CI on the
  annualized gap — i.i.d. resampling would understate uncertainty on a series with monthly
  clustering.
- **The FX-drag regression isolates the incremental currency channel.** Because *both* EMB and
  AGG already carry a negative beta to the dollar (shared macro/credit-cycle exposure — dollar
  strength is generically a risk-off signal), regressing each leg individually on UUP conflates
  currency with credit. Regressing the **Local-minus-EMB difference itself** on the dollar nets
  out the shared component and isolates what's unique to holding the local currency.
- **Named crisis windows are chosen ex ante** from the well-known EM-local-debt drawdown
  episodes the claim's own believers cite (the 2013 taper tantrum, the 2015 China-devaluation /
  commodity selloff, the 2022 Fed-hiking dollar surge) — not snooped from this study's own
  results.
- **No execution lag to document.** This is a static buy-and-hold exposure comparison, not a
  discretionary signal — the house convention exempts calendar-known, non-timed holdings from
  the "one documented lag" rule the same way a scheduled-calendar entry needs none. Costs are
  charged once, at entry, and shown explicitly even though a 14.7-year amortization makes them
  invisible at the reported precision.

## Data sources

- **EBND, LEMB, EMB, AGG, UUP, BIL** daily total-return-adjusted closes (`auto_adjust=True`) —
  yfinance (no key), cached under `_cache/` (`elb_prices.csv`), 2003-09-29 → 2026-06-30 (common
  sample from LEMB's 2011-10-20 inception: 2011-11-30 → 2026-06-30).
  - EBND: SPDR Bloomberg Emerging Markets Local Bond ETF — https://www.ssga.com/us/en/intermediary/etfs/spdr-bloomberg-emerging-markets-local-bond-etf-ebnd
  - LEMB: iShares J.P. Morgan EM Local Currency Bond ETF — https://www.ishares.com/us/products/239495/
  - EMB: iShares J.P. Morgan USD Emerging Markets Bond ETF — https://www.ishares.com/us/products/239572/
  - AGG: iShares Core U.S. Aggregate Bond ETF — https://www.ishares.com/us/products/239458/
  - UUP: Invesco DB US Dollar Index Bullish Fund — https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=UUP
  - BIL: SPDR Bloomberg 1-3 Month T-Bill ETF — https://www.ssga.com/us/en/intermediary/etfs/spdr-bloomberg-1-3-month-t-bill-etf-bil
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [612-em-debt-carry](../612-em-debt-carry/) — the **USD-denominated** (hard-currency) EM
  sovereign carry trade, EMB vs IEF. No currency leg at all; that study's whole point is to
  isolate the *credit* premium. This study is the currency leg 612 deliberately avoids.
- [364-fx-carry-trade](../364-fx-carry-trade/) — the classic **G10 FX carry trade** (long
  high-yield currencies, short low-yield funding currencies, spot FX only, no bonds). Same
  underlying economic force (UIP failure / currency risk premia) but a different instrument —
  spot currency positions, not local-currency *sovereign bonds*, and a G10, not EM, universe.
- [660-carry-everywhere](../660-carry-everywhere/) — the **Koijen-Moskowitz-Pedersen-Vrugt**
  multi-asset carry factor (FX + Treasury + equity-dividend + commodity-roll legs combined).
  Its FX leg is G10 carry (see 364, above), and it never isolates EM local-currency debt as its
  own leg. This study is a single-asset-class deep dive 660's diversified combo does not cover.
- [339-convertible-bonds](../339-convertible-bonds/) — unrelated asset class (US convertible
  bonds vs a stock/bond blend), included in the brief's dedup list only because it shares this
  study's methodological pattern: a claimed payoff shape tested and found **statistically real
  but backward**, both stamped `NONE` rather than a bespoke "real-negative" grade — the
  precedent this study's Signal stamp follows.

None of the siblings test whether **local-currency EM bonds' extra promised yield survives FX
depreciation** — that is this study's own axis.
