# References & literature map — Study 611 (mREIT Carry)

## The claim under test

- **The pitch.** *"Mortgage REITs pay 10-14% — leveraged MBS carry. Yes the price swings, but
  the income is real: harvest the dividend stream and the carry compounds."* The staple of
  income newsletters, dividend screens and "high-yield retirement portfolio" model books;
  NLY and AGNC are perennial top-of-screen names, and REM/MORT package the sector.
- **What an agency mREIT actually is.** A levered spread book: buy agency MBS (duration ~4-6y,
  negatively convex), fund it with short-dated **repo** at ~5-9× leverage, hedge part of the
  duration with swaps/swaptions, pay out ~90% of taxable income as required by REIT status.
  The dividend is the levered net interest margin — classic **carry**: short liquidity, short
  volatility, long the financing regime.

## Why the carry can be real while the vehicle is a mirage

- **Carry, generically.** Koijen, Moskowitz, Pedersen & Vrugt (2018, *Carry*, JFE): carry
  predicts returns in every asset class but is compensation for crash/liquidity risk — carry
  strategies earn steadily and lose violently. The mREIT coupon is a packaged, retail-facing
  instance.
- **Run-on-repo mechanics.** Gorton & Metrick (2012, *Securitized banking and the run on repo*,
  JFE) — the GFC mechanism that took REM down −74.7%. The March-2020 dash-for-cash reran it:
  see the Fed's *Financial Stability Report* (May 2020) sections on mREIT deleveraging, and
  Schrimpf, Shin & Sushko (BIS Bulletin No. 2, 2020) on margin spirals.
- **Negative convexity.** Hanson (2014, *Mortgage convexity*, JFE): MBS holders are short the
  homeowner's prepayment option, so rate moves in *either* direction hurt — the 2013 taper
  tantrum (−25 to −37% across our names while SPY fell −5.6%) is the signature.
- **Return of capital dressed as yield.** The same packaged-carry pathology the desk documented
  on MLPs and BDCs — see the sibling studies below.

## What we measure, and why this construction

- **Dividend component = total-return minus price-only.** yfinance daily closes downloaded
  twice (`auto_adjust=True` vs `False`; Yahoo's raw Close is split-adjusted but not
  dividend-adjusted), resampled to month-end. The monthly difference is the dividend return —
  the "carry" the pitch sells — with a **Newey-West HAC t** (payout streams are serially
  correlated; Newey & West 1987).
- **Duration-matched levered benchmark.** NW-HAC regression of the mREIT's monthly excess
  return (over BIL) on **IEF** and **SPY** excess returns. The intercept is the carry premium
  over the passive levered-Treasuries-plus-equity mix an investor could hold instead; full-
  sample betas are a benchmarking choice for a risk decomposition (stated openly), not a
  timed trading rule. Excess-on-excess throughout, so the alpha is a genuine risk-adjusted
  spread (Sharpe 1994 ratio logic).
- **Crisis autopsies on fixed calendar windows.** GFC (2007-06→2009-03), taper tantrum
  (2013-05→2013-12, Bernanke's 2013-05-22 testimony), COVID (2020-02→2020-04), rate shock
  (2022-01→2022-10) — documented dates, not fitted windows; peak-to-trough total-return
  drawdowns on the daily tape.
- **Survivorship.** REM is an index fund — the sector's blow-ups are inside its tape, so the
  headline leg is not survivor-biased. NLY/AGNC are the two biggest **survivors** and are
  quoted as colour only; this is named on the Signal axis.

## Data sources used here

- **yfinance** daily closes (total-return and price-only) for REM, NLY, AGNC, IEF, SPY, BIL,
  2007-06 → 2026-06, cached under `_cache/mrc_tr.csv` / `_cache/mrc_px.csv`. All headline
  numbers pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (as-of 2026-06-30, fingerprint `3b2ec2fc7dd3`).
- iShares REM fund page (expense ratio 0.48%, index methodology):
  https://www.ishares.com/us/products/239543/
- Koijen, Moskowitz, Pedersen & Vrugt (2018): https://doi.org/10.1016/j.jfineco.2017.11.002
- Gorton & Metrick (2012): https://doi.org/10.1016/j.jfineco.2011.03.016
- Hanson (2014): https://doi.org/10.1016/j.jfineco.2014.05.006
- Federal Reserve *Financial Stability Report*, May 2020:
  https://www.federalreserve.gov/publications/files/financial-stability-report-20200515.pdf

## Related desk studies (the packaged-carry family — the dedup frame)

- **[341 — MLP-Pipelines](../341-mlp-pipelines/)** (Real × Mirage): the "7-8% toll-road
  income" that is levered energy beta with a return-of-capital coupon.
- **[342 — BDC-Yield](../342-bdc-yield/)** (Real × Mirage): the "~10% senior-loan income"
  that is levered private-credit equity.
- **This study is the third sibling, new asset:** the *mortgage* flavour — levered **agency
  MBS repo carry**. Same family pathology (a real, harvestable coupon financed by NAV
  erosion, with the true beta hiding under the income label), but a distinct mechanism: here
  the coupon really is a **carry trade** (levered net interest margin, short liquidity/
  convexity), the drawdown driver is the **repo margin spiral**, and the natural benchmark is
  **duration-matched levered Treasuries** rather than an equity sector. Distinct from
  [57 — Yield-Trap](../57-yield-trap/) (single-stock dividend screens) and
  [337 — Covered-Call-ETF](../337-covered-call-etf/) (option-overlay income illusion).
