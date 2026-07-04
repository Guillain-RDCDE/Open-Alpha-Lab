# References — Study 599 (Tax-Loss Harvesting)

## The claim's source

- **Wealthfront — "Tax-Loss Harvesting" white paper** — the robo-advisor pitch that daily,
  automated TLH adds on the order of **1%+/yr of after-tax return** ("Tax-Alpha"), harvesting
  into a correlated twin ETF to stay invested.
  <https://research.wealthfront.com/whitepapers/tax-loss-harvesting/>
- **Betterment — "Tax Loss Harvesting+ methodology"** — the competing robo claim (~0.77%/yr
  advertised historically), same twin-fund mechanics.
  <https://www.betterment.com/legal/tax-loss-harvesting-methodology>
- The generic advisor folklore version — *"systematic TLH adds 0.3–1%/yr of after-tax alpha"* —
  is the claim we test at full strength on a single broad-index (SPY↔IVV/VTI) implementation.

## Key papers

- **Chaudhuri, Burnham & Lo (2020), "An Empirical Evaluation of Tax-Loss-Harvesting Alpha",
  *Financial Analysts Journal* 76(3)** — the canonical academic estimate: ~**1.08%/yr** of
  tax alpha 1926–2018 on a *stock-level* (25-stock) portfolio with monthly harvesting; they
  stress that the alpha is front-loaded, path-dependent and rate-dependent.
  <https://doi.org/10.1080/0015198X.2020.1760064>
- **Arnott, Berkin & Ye (2001), "Loss Harvesting: What's It Worth to the Taxable Investor?",
  *Journal of Wealth Management* 3(4)** — Monte-Carlo estimate of ~**50–60 bps/yr**, decaying
  as the portfolio's basis locks up.
  <https://doi.org/10.3905/jwm.2001.320407>
- **Constantinides (1983), "Capital Market Equilibrium with Personal Tax", *Econometrica* 51(3)**
  — the theory: with taxes on *realisation*, the optimal policy is to realise losses immediately
  and defer gains — TLH is the practical corollary.
  <https://doi.org/10.2307/1912156>
- **Berkin & Ye (2003), "Tax Management, Loss Harvesting, and HIFO Accounting", *Financial
  Analysts Journal* 59(4)** — lot-level accounting (HIFO) and the decay of harvesting
  opportunities over the portfolio's life. <https://doi.org/10.2469/faj.v59.n4.2544>
- **Israel & Moskowitz (2012), "How Tax Efficient Are Equity Styles?"** — tax-aware
  implementation context for passive/style portfolios.
  <https://doi.org/10.2139/ssrn.2089459>

## Rules & mechanics

- **IRC §1091 (wash-sale rule)** — a loss is disallowed if a *substantially identical* security
  is bought within 30 days before or after the sale; the industry treats two different-issuer
  index ETFs (SPY↔IVV/VTI) as *not* substantially identical (never formally blessed by the
  IRS — a named implementation risk). <https://www.law.cornell.edu/uscode/text/26/1091>
- **IRC §1(h) / §1211–1212** — ST vs LT rate schedule, the $3,000 ordinary-income offset cap
  and indefinite loss carry-forward (our "fully usable losses" assumption brackets the cap).
- **IRC §1014 (step-up in basis at death)** — deferral becomes forgiveness; the `step_up`
  switch in the engine. <https://www.law.cornell.edu/uscode/text/26/1014>

## Data

- **SPY & IVV daily total-return closes** — Yahoo! Finance via `yfinance`
  (`auto_adjust=True`), SPY 1993-01-29 →, IVV 2000-05-19 → (twin-equivalence check only).
  Cached under `_cache/`. <https://pypi.org/project/yfinance/>

## Named siblings (household-finance folklore family — distinct questions)

- [Study 101 — Slow-and-Steady](../101-slow-and-steady/) — DCA vs lump sum (deployment timing).
- [Study 102 — Free-Rebalance](../102-free-rebalance/) — the "rebalancing bonus" free lunch.
- [Study 172 — Hundred-Minus-Age](../172-hundred-minus-age/) — the age-based glidepath rule.
- [Study 173 — Four-Percent-Rule](../173-four-percent-rule/) — the retirement withdrawal rule.
- This study is the family's **pure tax-arithmetic** member: no market-timing claim at all —
  the twin swap keeps exposure identical and the entire delta is tax timing and rate arbitrage.

## Shared method citations

- **Newey & West (1987)** — HAC standard errors for the overlapping-cohort mean.
- **Welch (1947)** — unequal-variance t for the bear-vs-calm harvest-yield split.
