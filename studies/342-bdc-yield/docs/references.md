# References & literature map — Study 342 (BDC-Yield)

## The claim under test

**Business development companies (BDCs)** — and the **BIZD** ETF (VanEck BDC Income ETF)
that bundles them — are sold on a single number: a **~10% distribution yield**. The pitch
is that BDCs hold *senior secured loans to private mid-market companies*, so you collect a
double-digit income "with the safety of senior debt," a high-income complement to bonds.
Income-investing media, dividend communities and the funds' own marketing lead with the
headline yield.

- VanEck's positioning for **BIZD** (*VanEck BDC Income ETF*, launched 2013-02-11/12) and
  its peers (e.g. the Putnam/Amplify analogues) foreground the high distribution rate.
- The retail framing ("get ~10% income from senior loans") is ubiquitous in
  high-yield / income-investing media.

## Why the steelman is almost coherent

- **Seniority is real on the loans.** Most BDC assets are first-lien / senior secured
  floating-rate loans, which *do* rank ahead of the borrowers' equity and subordinated
  debt. On paper that is a senior credit exposure.
- **The distribution is real and high.** BDCs are regulated investment companies that must
  distribute ~90% of taxable income, and floating-rate loan yields are genuinely high — so
  the ~10% headline is not invented.

## Why it fails *as stated* ("safe ~10% income")

- **A BDC is a leveraged equity, not a bond.** The *fund* (BDC) is the equity tranche
  sitting on top of its own borrowing — it runs balance-sheet leverage (regulatory debt-to-
  equity up to ~2:1 since the 2018 SBCAA), is externally managed with fat fees, and trades
  at a market price that swings with sentiment and the discount/premium to NAV. The ETF
  (BIZD) is a basket of *those* equities. Senior loans underneath ≠ senior exposure for the
  holder.
- **The borrowers are the riskiest part of the credit market.** BDC loans fund private,
  non-investment-grade, often PE-owned mid-market companies — exactly the credits that
  default in a downturn. A levered claim on sub-IG private credit carries a deep
  credit-spread / equity-tail exposure.
- **The 2020 record is the canonical counter-example.** Leveraged-loan and BDC vehicles
  fell ~50%+ in the COVID crash — *more than the S&P 500* — while Treasuries rallied, the
  opposite of the "safe income" the headline implies.
- **The distribution is partly a return of capital.** A high quoted distribution rate on a
  vehicle whose NAV drifts down over time means a chunk of the "yield" is the fund handing
  back your own capital; the realised *total* return is far below the headline.

## Method lineage

- **Univariate / downside beta.** Conditioning beta on down-market days follows the
  downside-risk tradition (Bawa & Lindenberg 1977; Ang, Chen & Xing, *Downside Risk*, RFS
  2006) — the relevant measure when the claim is about behaviour *in the crash*.
- **Newey–West HAC standard errors** for the mean of an autocorrelated influence series
  (here, the OLS-beta influence function): Newey & West (1987), Econometrica.
- **Circular block bootstrap** for a CI on a beta *difference* — block resampling preserves
  volatility clustering and cross-asset co-movement that i.i.d. resampling destroys
  (Politis & Romano, 1994).
- **Distribution-rate vs total-return distinction.** A quoted distribution yield includes
  return of capital and ignores NAV change; the fair measure of what an investor earned is
  the dividend-and-split-adjusted total return (`yfinance auto_adjust=True`). A price-only
  series would understate BIZD's gross return; the headline yield overstates the net one.

## Data sources used

- **BIZD** (VanEck BDC Income ETF), **SPY** (equity), **IEF** (7-10y Treasuries), daily,
  **total-return adjusted** via `quantlab.data` (yfinance `auto_adjust=True`); SPY/IEF from
  the shared cross-asset cache, BIZD cached study-local. BIZD lists **2013-02-12**, which
  bounds the joint window honestly — stated as a decision, not buried.

## Related desk studies

- [Study 338 — Preferred-Stocks](../../338-preferred-stocks/) — "bond-like safety with an
  equity yield?" Same crash-co-movement lens. **Distinct**: preferreds are an *unlevered*
  junior security; BDCs are *externally-managed, balance-sheet-levered* private-credit
  equity sold on a headline *distribution rate*.
- [Study 340 — Bank-Loans](../../340-bank-loans/) — senior leveraged loans (BKLN).
  **Distinct**: BKLN holds the loans directly; BIZD holds the *levered fund equity* that
  sits on top of them — a second layer of leverage and fees.
- [Study 339 — Convertible-Bonds](../../339-convertible-bonds/) and
  [Study 207 — REITs-Diversifier](../../207-reits-diversifier/) — other "is this asset what
  the brochure says?" income/hybrid teardowns.
- [Study 97 — Balancing-Act](../../97-balancing-act/) — the fixed 60/40 baseline an income
  sleeve is implicitly compared against.
