# References & literature map — Study 618 (GBTC Premium Cycle)

## The claim under test

- **The lifecycle itself.** Grayscale Bitcoin Trust (symbol GBTC): private placement opens
  2013-09-25 at **0.1 BTC per share**; OTCQX quotation approved 2015-05-04 (Grayscale press /
  OTC Markets); trades at a large, persistent **premium** to held bitcoin 2015–2021; flips to a
  persistent **discount** in Feb-2021 that bottoms near **−49 %** in Dec-2022; converts to a
  spot **ETF on NYSE Arca on 2024-01-11** (SEC approval order 34-99306, 2024-01-10) and the
  premium collapses to ≈ 0. The regime dates used here are all from this public record.
- **Why a premium could exist at all.** Until late 2017 (CME futures) and effectively 2024
  (spot ETFs), GBTC was the only way to hold bitcoin exposure in a US brokerage/IRA account.
  Shares could be **created at NAV only by accredited investors** (Reg D private placement,
  $50k minimum) and resold publicly only after the **Rule 144 lockup** (12 months; 6 months
  after the 2019 amendments) — and the trust had **no redemption mechanism** (Grayscale shut
  its redemption program in 2014 after SEC Rule 102 proceedings, In re Genesis Global Trading /
  SEC 34-77872 background). One-way creation + retail demand = premium; competition + no
  redemption = discount. The mechanics are spelled out in every GBTC 10-K (SEC EDGAR,
  CIK 0001588489).

## Trust mechanics used in the BTC-per-share model (hardcoded in `data.py`)

- **Inception ratio.** 0.1 BTC/share at the 2013-09-25 launch — GBTC Form 10/annual reports.
- **Sponsor fee 2.0 %/yr, accrued daily in BTC** while a trust; **1.5 %/yr** after the ETF
  conversion (Grayscale press release, 2024-01-08). BTC-per-share therefore decays ~exp(−fee·t).
- **91-for-1 split**, record 2018-01-22, effective 2018-01-26 (Grayscale announcement). Yahoo
  prices are split-adjusted throughout; the model works in post-split shares (0.1/91) from
  inception. Model check: **0.0010076** BTC/share at the split (Grayscale disclosed ≈0.00101),
  **0.00089442** at conversion (Grayscale disclosed ≈0.00089 — ~619k BTC / ~692M shares).
- **Grayscale Bitcoin Mini Trust spin-off, 2024-07-31**: GBTC distributed **10 %** of its
  bitcoin to shareholders as Mini-Trust (NYSE: BTC) shares (Grayscale press release,
  2024-07-31). Yahoo folds this into all historical price columns as a ×0.90 back-adjustment
  with no dividend row, so the same ×0.90 factor applies to the whole modeled BTC-per-share
  path. **Self-validation:** the ETF-era premium — pinned to ≈0 by the in-kind create/redeem
  arb — reads mean −0.026 % on our reconstruction.
- Minor omission, stated: the trust's 2017–18 fork/airdrop dispositions (e.g. Bitcoin Cash)
  produced one small cash distribution (<1 % of NAV) that Yahoo does not carry; it is inside
  our ±0.9 % ETF-era residual noise.

## The dated catalysts (event study)

- **2023-06-15 — BlackRock files Form S-1 for the iShares Bitcoin Trust** (SEC EDGAR,
  CIK 0001980994; filed after hours). The first credible spot-ETF sponsor: the discount
  narrowed ~9 log-pts at the next close (z ≈ +3.5) and the convergence began in earnest.
- **2023-08-29 — *Grayscale Investments, LLC v. SEC*, No. 22-1142 (D.C. Cir.)** — the court
  vacates the SEC's denial of GBTC's ETF conversion as arbitrary and capricious. One-day
  hedged move +9.64 % (z ≈ +3.7).
- **2024-01-10 — SEC Release 34-99306** approves the spot bitcoin ETPs; **2024-01-11** GBTC
  uplists to NYSE Arca. By then the discount was −2.5 % — the terminal event was priced.

## Related literature

- **Closed-end fund discounts.** Lee, Shleifer & Thaler (1991, *Investor Sentiment and the
  Closed-End Fund Puzzle*, JF); Pontiff (1996, *Costly Arbitrage: Evidence from Closed-End
  Funds*, QJE) — a fund without redemption can trade far from NAV, bounded only by the cost of
  the arb. GBTC is the extreme case: the arb (creation) was one-way, gated and locked up, so
  the bound was extremely loose — ±50 % realized.
- **The GBTC premium/discount specifically.** Ammous/academic and practitioner post-mortems of
  the 2020–21 "GBTC carry trade" (create-at-NAV, dump-at-premium) and its role in the Three
  Arrows Capital / BlockFi / Genesis failures — see the 3AC bankruptcy filings (In re Three
  Arrows Capital, SDNY 22-10920) and public reporting; the trade's mechanics match our cohort
  arithmetic (§ third axis).
- **ETF arbitrage as the premium-killer.** Petajisto (2017, *Inefficiencies in the Pricing of
  Exchange-Traded Funds*, FAJ) — in-kind create/redeem keeps ETF price-NAV gaps to basis
  points; exactly the regime GBTC entered on 2024-01-11 (our ETF-era mean |premium| = 0.64 %).
- **Newey & West (1987)** — HAC t used on the convergence drift and (with a long 63-day lag,
  flagged as supporting-only) on the regime levels. **Welch (1947)** for the regime split.

## Data sources

- **yfinance**: GBTC daily closes (OTCQX then NYSE Arca; split/spin-off-adjusted) and BTC-USD
  daily closes, 2015-05-11 → 2026-06-30, cached under `_cache/gbtc.csv` / `_cache/btc.csv`.
  BTC-USD closes stamp ~00:00 UTC vs GBTC's 16:00 ET — noted as noise, not bias.
- Trust mechanics: Grayscale press releases + GBTC filings on SEC EDGAR (CIK 0001588489), URLs
  in the docstrings of [`gbtc_premium_cycle/data.py`](../gbtc_premium_cycle/data.py).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (dedup)

- [324-bitcoin-treasury](../324-bitcoin-treasury/) — **MSTR**: bitcoin wrapped in an operating
  company, the *equity-premium-on-wrapped-BTC* story. This study is the **fund wrapper's arb
  lifecycle** — creation gates, lockups, no redemption, then in-kind arb — a different animal:
  GBTC's premium died by mechanical conversion, MSTR's floats on leverage and narrative.
- [70-digital-gold](../70-digital-gold/) — bitcoin itself as an asset; here BTC is only the
  hedge leg.
