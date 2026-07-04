# References & literature map — Study 614 (CLO Equity Yield — "the 15% Machine")

## The claim under test

- **The pitch.** *"ECC and OXLC pay ~15%+ distributions from CLO equity. A CLO's equity
  tranche collects the excess spread of a whole leveraged-loan portfolio after the debt
  tranches are paid — 15-25% cash-on-cash in normal years. The funds pass it to you monthly.
  The income is real."* A staple of high-yield income newsletters, Seeking Alpha's
  "retire on 15%" genre, and the CEF-income community; OXLC is routinely among the
  highest-yielding listed funds in the US.
- **What a CLO-equity CEF actually is.** Double leverage on first-loss credit: a CLO equity
  tranche is the *residual* of a ~10× levered loan portfolio (it absorbs the first defaults
  and the managers' fees), and ECC/OXLC then add fund-level leverage (preferred shares,
  notes) plus management (~1.75-2.5% of *gross* assets), incentive fees and leverage
  interest — total expense ratios in the funds' own reports run **high-single to
  low-double-digit percent of NAV per year**. The distribution is set by the board, not by
  earnings; the difference is the NAV.

## Why the payout can be real while the vehicle is a mirage

- **Carry, generically.** Koijen, Moskowitz, Pedersen & Vrugt (2018, *Carry*, JFE): carry
  earns steadily and crashes violently — compensation for crash/liquidity risk. CLO equity
  is credit carry at its most concentrated: the first-loss tranche of a levered loan book.
- **CLO economics.** Cordell, Roberts & Schwert (2023, *CLO Performance*, Journal of
  Finance): CLO *equity tranches* held to maturity earned positive but highly dispersed
  returns; the economics accrue to managers via fees in the weaker vintages. The listed
  wrappers stack fund-level fees and leverage on top of that.
- **Distribution ≠ earnings.** Closed-end fund distributions are board-declared; SEC Rule
  19a-1 requires funds to disclose when a distribution includes a **return of capital** —
  ECC's and OXLC's 19(a) notices have repeatedly flagged exactly that. Our returns-arithmetic
  split (share of the payout offset by price erosion) is the market-price mirror of those
  notices.
- **Return of capital dressed as yield.** The same packaged-carry pathology the desk
  documented on bank-loan funds, MLPs, BDCs and mREITs — see the sibling studies below.

## What we measure, and why this construction

- **Distribution component = total-return minus price-only.** yfinance daily closes
  downloaded twice (`auto_adjust=True` vs `False`; Yahoo's raw Close is split-adjusted but
  not distribution-adjusted — OXLC's 2025-09 1-for-5 reverse split is handled on both legs),
  resampled to month-end. The monthly difference is the distribution return — the "15%" the
  pitch sells — with a **Newey-West HAC t** (payout streams are serially correlated; Newey
  & West 1987).
- **Headline test: TR spread vs HYG.** The fund's monthly total return minus HYG's, HAC t.
  HYG is the plain, unlevered, one-click credit alternative sitting in the same brokerage
  account — the realistic opportunity cost for the income investor the pitch targets.
- **Credit/equity-matched benchmark.** NW-HAC regression of the fund's monthly excess
  return (over BIL) on **HYG** and **SPY** excess returns. The intercept is the premium over
  a passive levered-credit-plus-equity mix; full-sample betas are a benchmarking choice for
  a risk decomposition (stated openly), not a timed trading rule. Excess-on-excess
  throughout (Sharpe 1994 ratio logic).
- **Third axis (returns arithmetic).** Share of the annualised distribution stream offset
  one-for-one by the price-only CAGR — 0% means the payout rode on a flat price (return ON
  capital); 100% means the payout was fully matched by capital shrinkage (return OF capital).
- **Crisis autopsies on fixed calendar windows.** 2015-16 credit crunch (2015-06→2016-02),
  Q4 2018 loan-fund outflows (2018-09→2018-12), COVID (2020-02→2020-04), rate shock 2022
  (2022-01→2022-10) — documented dates, not fitted windows; peak-to-trough total-return
  drawdowns on the daily tape.
- **Survivorship.** There is **no index fund for CLO equity**: the listed category
  essentially *is* ECC and OXLC, the two wrappers that scaled and survived to 2026. The
  panel is survivor-tilted by construction; named on the Signal axis.

## Data sources used here

- **yfinance** daily closes (total-return and price-only) for ECC, OXLC, HYG, SPY, BIL,
  2010-01 → 2026-06, cached under `_cache/ceq_tr.csv` / `_cache/ceq_px.csv`. All headline
  numbers pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (as-of 2026-06-30, fingerprint
  `366d9cf7d356`).
- Eagle Point Credit Co (fees, leverage, 19(a) notices): https://www.eaglepointcreditcompany.com/
- Oxford Lane Capital (fees, leverage, 19(a) notices, the 2025 reverse split):
  https://www.oxfordlanecapital.com/
- SEC Rule 19a-1 (return-of-capital disclosure): https://www.ecfr.gov/current/title-17/chapter-II/part-270/section-270.19a-1
- Cordell, Roberts & Schwert (2023), *CLO Performance*: https://doi.org/10.1111/jofi.13224
- Koijen, Moskowitz, Pedersen & Vrugt (2018): https://doi.org/10.1016/j.jfineco.2017.11.002
- Newey & West (1987): https://doi.org/10.2307/1913610

## Related desk studies (the packaged-carry family — the dedup frame)

- **[340 — Bank-Loans](../340-bank-loans/)**, **[341 — MLP-Pipelines](../341-mlp-pipelines/)**
  and **[342 — BDC-Yield](../342-bdc-yield/)** (all Real × Mirage/Fragile): the desk's
  packaged-carry family — a real, harvestable coupon financed by NAV erosion, with the true
  beta hiding under the income label.
- **[611 — mREIT-Carry](../611-mreit-carry/)** (Real × Mirage): the mortgage flavour, where
  the benchmark is duration-matched levered Treasuries.
- **This study is the loan-securitization flavour, and the most extreme of the family:**
  the coupon is the **residual cash flow of a first-loss tranche** (not a margin, not a
  toll), the headline yield is the family's biggest (17-19%/yr *measured*), the
  capital-consumption share is the starkest (~75-78% of the payout offset by price erosion),
  and the wrapper adds a **second layer of leverage and fees** on top of the CLO's own ~10×.
  Distinct from [342 — BDC-Yield](../342-bdc-yield/) (BDCs *originate* loans at ~1-1.3×
  leverage; here the funds *buy residual tranches* of ~10× levered pools) and from
  [340 — Bank-Loans](../340-bank-loans/) (the unlevered loan asset class itself).
