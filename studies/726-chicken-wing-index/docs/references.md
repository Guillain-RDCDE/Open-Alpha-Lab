# References & literature map — Study 726 (Chicken-Wing-Index)

## The claim and its demand story

- **National Chicken Council (NCC)** — *Chicken Wing Report* (annual, pre-Super-Bowl). The origin of the
  headline the whole trade rests on: Americans are projected to eat **~1.4 billion chicken wings** over
  Super Bowl weekend (1.45 bn for SB LVII/2023; 1.47 bn for 2025). This is the steelman — a genuine,
  enormous, calendar-locked demand pulse — that this study tears apart as a *tradable* signal.
- **Wingstop Inc. (NASDAQ: WING)** — investor materials and 10-K/10-Q filings. The pure-play US wing
  chain (IPO June 2015), routinely cited in business press as *the* Super-Bowl stock; its own IR notes the
  Super Bowl as one of its highest-volume days. The tradable leg this study actually tests.
- **Business/finance press "Super Bowl stock" seasonality pieces** (Barron's, MarketWatch, Bloomberg,
  Nasdaq/Zacks "buy Wingstop before the big game" notes) — the recurring retail-facing version of the
  claim: buy the wing stock into the game. It is exactly the folklore the desk exists to test.

## Wholesale wing-price dynamics (the physical leg — a labelled proxy)

- **USDA Agricultural Marketing Service (AMS)** — *National Poultry Market: Weekly/Daily* wholesale price
  sheets (whole/jumbo chicken wings, US$/lb). The public backbone of the hardcoded, approximate wing-price
  series in `data.load_wing_price` (a *proxy*, never a live feed).
- **Urner Barry** poultry quotes as echoed in trade and general press — the source most cited for the
  **record wing-price spike into early 2021** (pandemic wing shortage; wings briefly the priciest cut of
  the bird) and the subsequent **2022–2023 collapse**. Documents that wing prices are driven by
  **supply shocks (HPAI/avian influenza, processing bottlenecks)**, not by a smooth annual Super-Bowl bump
  — i.e. the price action is **event-driven, not calendar-driven**, which is why a fixed-month rule fails.
- **USDA APHIS** — *Highly Pathogenic Avian Influenza (HPAI) detections* — the tail-risk supply shock that
  dominates poultry-price variance, the poultry analogue of the storage/stockout dynamics below.

## On calendar effects, data-snooping and the January effect

- **Sullivan, R., Timmermann, A., & White, H. (2001).** *Dangers of Data Mining: The Case of Calendar
  Effects in Stock Returns.* Journal of Econometrics 105(1), 249–286 — the direct warning that a "best
  month" found by searching all twelve rarely survives a multiple-testing / Reality-Check correction. The
  spine of this teardown (here: November out-ranks the on-thesis January).
- **Rozeff, M. S., & Kinney, W. R. (1976).** *Capital Market Seasonality: The Case of Stock Returns.*
  Journal of Financial Economics 3(4), 379–402 — the canonical **January effect** documentation; the
  confound this study must net out before attributing WING's January to wings (WING is a smaller, high-beta
  growth name that would inherit any turn-of-year small-cap tailwind).
- **Haugen, R. A., & Lakonishok, J. (1988).** *The Incredible January Effect.* — the book-length treatment
  of exactly the seasonal WING's January could be riding rather than any Super-Bowl demand.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return Predictability?*
  Journal of Finance 71(1), 5–32 — post-publication decay applies to seasonal patterns as much as to
  factor premia; a widely-touted "buy the wing stock" trade is a prime decay candidate.

## Commodity-price / demand-shock theory

- **Deaton, A., & Laroque, G. (1992).** *On the Behaviour of Commodity Prices.* Review of Economic
  Studies 59(1), 1–23 — the storage model: perishable/agricultural prices are dominated by stockout-driven
  spikes (fat right tails), not smooth seasonal cycles, so seasonal means are swamped by tail events (the
  2021 wing spike is the textbook case).

## Shared method (the desk engine)

- **Newey, W. K., & West, K. D. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708 — the HAC t reported per
  month (and flagged as unreliable at n ≈ 11).
- **Lo, A. W. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52 — the
  autocorrelation-robust Sharpe SE used across the desk.
- **Politis, D. N., & Romano, J. P. (1994).** *The Stationary Bootstrap.* JASA 89(428), 1303–1313 — the
  block-bootstrap family behind the 95% CI on the window-minus-rest spread (12-month blocks to respect the
  annual seasonal structure).

## Data

- **Yahoo! Finance** — Wingstop (WING), the S&P 500 ETF (SPY) and the 13-week T-bill (^IRX, the cash leg),
  2015–2026, **daily** closes resampled to month-end (131 months). WING pays no dividend → effectively
  price-only; SPY is dividend-reinvested (total-return-ish). The study-local cache lives at
  `_cache/chicken_wing_index.parquet` (gitignored).
- **Wholesale wing price** — hardcoded, cited, **approximate** annual pre-Super-Bowl whole-wing price
  (US$/lb) in `chicken_wing_index/data.py`, reconstructed from the USDA/Urner-Barry/NCC reporting above. A
  **labelled proxy**, not a live feed, and **not directly tradable** (no wing futures market).

*Companion studies on the bench: [307 Coffee-Seasonality](../../307-coffee-seasonality/) (the frost/harvest
calendar), [358 Watch-Index](../../358-watch-index/) (a collectible "asset class" with a labelled proxy
series), and [708 Eurovision-Effect](../../708-eurovision-effect/) — the consumer-oddity / seasonality
family.*
