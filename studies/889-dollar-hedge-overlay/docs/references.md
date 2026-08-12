# References & literature map — Study 889 (Broad Dollar-Hedge Overlay)

## The claim under test

- **The parent result.** Our own [study 613 — Currency-Hedged-ETF-Carry](../../613-currency-hedged-etf-carry/)
  showed that for **one** market (Japan) the return gap between a currency-*hedged* equity ETF
  (HEWJ) and its *unhedged* twin (EWJ) is, mechanically, the covered-interest-parity short-rate
  differential — the hedge sells the foreign currency forward, CIP prices that forward at
  `r_US − r_JP`, and the wrapper pockets it. This study asks: **does the identity generalise to a
  broad developed-international basket** (MSCI EAFE), where the differential is now *positive*
  (dollar-favourable) rather than the Japan-specific carry?
- **Covered interest parity.** The forward premium on a currency equals the interest-rate
  differential (Keynes 1923; see any international-finance text, e.g. **Sarno & Taylor, *The
  Economics of Exchange Rates*, 2002**). A one-month currency-hedged fund rolls one-month forwards,
  so its return differential vs the unhedged fund is (to first order) `(r_US − r_foreign)/12 − fx`
  each month, i.e. `carry_hat := (hedged − unhedged) + fx_foreign ≈ (r_US − r_foreign)/12`.
- **What "broad" changes.** Across developed-ex-US the foreign short rate is a *basket* (ECB, BOJ,
  BOE, SNB, …). Since 2022 the Fed has sat above most of them, so the hedge now *pays* the US holder
  the differential — the opposite sign to the classic negative-carry hedged-Japan story. We test the
  identity and a "hedge when the US out-yields" overlay.

## What we measure, and the honesty rails

- **The clean same-basket pair.** HEFA (iShares Currency Hedged MSCI EAFE) literally *holds EFA plus
  one-month currency forwards*, so `HEFA − EFA` is the hedge P&L almost purely — the broad analogue
  of 613's decisive HEWJ/EWJ pair. DBEF/EFA (same index, different provider) adds a longer,
  basket-noisier corroboration.
- **Spot, not UUP, for the carry.** `carry_hat = diff + fx_foreign` needs the pure **spot** USD
  return of the foreign basket, built here from EUR/JPY/GBP/CHF spot at EAFE-ish weights. `UUP`'s
  *total* return also earns the US-bill collateral yield, so `diff − UUP` cancels most of the carry
  — a documented trap, kept in the notebook as a worked example.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly carry mean and on the
  hedge regression `diff = α + β·(−fx_foreign)` — hedge-roll timing induces serial correlation, so a
  plain *t* would overstate significance. A block-bootstrap CI on the carry mean and on each sleeve's
  excess-of-cash Sharpe; a 2022 era split; a 20-seed synthetic positive control.
- **Costs and a documented lag.** The overlay reads the *prior* month-end differential (observable
  policy rates) to choose the class held next month; a full switch costs 2 × one-way bps. The
  isolation spread pays 50 bps/yr borrow. Excess-of-cash Sharpes subtract the BIL monthly return
  from both legs (an excess-vs-excess race).
- **Young-ETF / one-regime caveat, named on the Signal axis.** HEFA's clean tape starts 2014-03 and
  the whole sample sits in a single US-out-yields / rising-dollar era; the regime-independent
  evidence is the carry identity, not the realised Sharpe gap.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* on the carry mean and the hedge regression).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap (the Sharpe and
  mean CIs).
- **Lo, A. (2002)** — the statistics of Sharpe ratios (why a |t| < 2 Sharpe is not distinguishable
  from zero, however pretty the curve).

## Data sources

- **yfinance monthly total-return closes** (`auto_adjust=True`): HEFA, EFA, DBEF, IEFA, UUP, BIL,
  `^IRX`, and EUR/JPY/GBP/CHF spot, 2011-01 → 2026-06, cached under `_cache/`.
- **Policy-rate step tables** (hardcoded, sources in `data.py`): ECB deposit facility
  (ecb.europa.eu), BOJ overnight call-rate target (boj.or.jp), BoE Bank Rate (bankofengland.co.uk),
  SNB policy rate (snb.ch) — blended EAFE-weighted.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [613-currency-hedged-etf-carry](../../613-currency-hedged-etf-carry/) — the **single Japan pair**
  identity (HEWJ/EWJ, HEDJ/VGK). This study **generalises** it to a broad developed-ex-US EAFE
  basket with a *positive, dollar-favourable* differential, and tests a dollar-regime hedge overlay —
  not one country's negative carry.
- [114-dollar-smile](../../114-dollar-smile/) — the dollar's *macro* behaviour (the "smile": the USD
  rallies in both global booms and busts). This study is a *mechanical wrapper carry*, not a
  directional dollar-macro forecast.
- [828-fx-dollar-factor](../../828-fx-dollar-factor/) — a **cross-currency dollar risk factor** (the
  common dollar component across FX pairs). This study takes no cross-currency positions; it measures
  the hedge P&L inside an equity share class.
- [145-home-bias](../../145-home-bias/) — whether US investors *should* hold more international. This
  study is agnostic on the equity allocation; it asks only what the *currency hedge* mechanically
  adds once you already hold EAFE.

None of the siblings measure the **broad-EAFE hedged-minus-unhedged return as the CIP differential**
— this study's own axis.
