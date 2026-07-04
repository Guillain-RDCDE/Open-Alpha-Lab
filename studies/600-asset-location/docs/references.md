# References — Study 600 (Asset Location)

## The claim's source

- **Dammon, R. M., Spatt, C. S., & Zhang, H. H. (2004).** *Optimal Asset Location and
  Allocation with Taxable and Tax-Deferred Investing.* **Journal of Finance**, 59(3), 999–1037.
  <https://doi.org/10.1111/j.1540-6261.2004.00655.x> — the canonical result: hold the
  tax-inefficient asset (taxable bonds) in the tax-deferred account, and locate equity in the
  taxable account, "a strong locational preference" worth meaningful after-tax wealth.
- **Shoven, J. B., & Sialm, C. (2004).** *Asset Location in Tax-Deferred and Conventional
  Savings Accounts.* **Journal of Public Economics**, 88(1–2), 23–38.
  <https://doi.org/10.1016/S0047-2727(02)00083-5> — the companion result; also shows the
  preference can reverse for high-turnover (tax-inefficient) equity funds — our third axis.
- **Popular form of the claim** — Bogleheads wiki, *Tax-efficient fund placement*:
  <https://www.bogleheads.org/wiki/Tax-efficient_fund_placement> — "put tax-inefficient assets
  (bonds) in tax-advantaged accounts"; the retail phrasing we test verbatim.

## Key papers & context

- **Poterba, J. M., & Samwick, A. A. (2003).** *Taxation and Household Portfolio Composition:
  US Evidence from the 1980s and 1990s.* **Journal of Public Economics**, 87(1), 5–38 — how
  households actually locate assets (often pro-rata, our default-policy benchmark).
- **Bergstresser, D., & Poterba, J. (2004).** *Asset Allocation and Asset Location: Household
  Evidence from the Survey of Consumer Finances.* **Journal of Public Economics**, 88(9–10),
  1893–1915 — most households violate the rule; the delta we measure is what they leave behind.
- **Sialm, C., & Zhang, H. (2020).** *Tax-Efficient Asset Management: Evidence from Equity
  Mutual Funds.* **Journal of Finance**, 75(2), 735–777 — equity-fund tax burdens vary hugely
  with turnover; the knob our third axis sweeps.
- **Vanguard Research (2022).** *Asset Location for Taxable Investors* — practitioner estimate
  of 5–30 bps/yr of "tax alpha" from location; our long-tape +17.8 bps/yr sits inside it.

## Sibling studies (the personal-finance arithmetic family)

This study grades a *mechanical* after-tax delta, like its siblings — none of which covers
asset **location**:

- [Study 101 — slow-and-steady](../../101-slow-and-steady/) (DCA arithmetic),
- [Study 156 — martingale](../../156-martingale/) (position-sizing arithmetic),
- [Study 172 — hundred-minus-age](../../172-hundred-minus-age/) (glidepath allocation folklore),
- [Study 173 — four-percent-rule](../../173-four-percent-rule/) (withdrawal-rate arithmetic),
- [Study 599 — tax-loss-harvesting](../../599-tax-loss-harvesting/) (the other "free tax alpha"
  claim; same lot). Study 600 is the *placement* question those studies never ask: same funds,
  same 60/40, different account.

## Data sources

- **Shiller long tape** — Robert Shiller, *Irrational Exuberance* data (monthly S&P composite
  price, dividend, GS10 long rate), <http://www.econ.yale.edu/~shiller/data.htm>, via the
  `datasets/s-and-p-500` mirror:
  <https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv> (cached at
  `_cache/al_shiller.csv`).
- **Modern tape** — Yahoo! Finance via `yfinance` (no key): SPY and IEF unadjusted closes +
  cash distributions, <https://finance.yahoo.com/> (cached at `_cache/al_modern_prices.csv`,
  `_cache/al_modern_divs.csv`).
- **US tax parameters** — IRS Rev. Proc. 2023-34 / 2024-40 bracket structure (ordinary brackets
  22/24/32/37%, LTCG+QDI 15/20%); the 20% LTCG threshold sits inside the top ordinary brackets,
  which is why the negative grid cells are not realistic household pairs.

## Shared method citations

- **Newey, W. K., & West, K. D. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.* **Econometrica**, 55(3), 703–708 — the HAC
  *t* (lag = horizon − 1 = 29) on overlapping-cohort means and the delta-on-yield regression.
- Desk-wide rules: [METHODOLOGY.md](../../../METHODOLOGY.md) — the inference bar, the one-lag
  rule, the synthetic-control discipline.
