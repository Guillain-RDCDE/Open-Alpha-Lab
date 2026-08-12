# References & literature map — Study 899 (Cash + Call "90/10")

## The claim under test

- **The source rule.** Bill **Gross** (PIMCO) popularised a **"90/10"** capital-protection portfolio
  in his *Investment Outlook* commentaries: keep the great majority of capital in short T-bills so it
  accretes back toward par (principal is roughly protected) and spend the coupon / a small ~10% slice
  on **convex upside** — call options — so a rally is captured with leverage while a crash costs only
  the premium. The idea is the retail cousin of a **protective / 90-10 structured note**.
- **The academic root.** Zvi **Bodie**, *"On the Risk of Stocks in the Long Run"* (Financial Analysts
  Journal, 1995) and related work frame the same construction — **T-bills plus a call option** — as
  the clean way to obtain equity upside with a hard floor on capital, and stress that the *cost* of
  that floor (the option premium) is precisely the price of the protection, not a free lunch: the
  put/call that insures a long horizon is *expensive*, and gets more so with volatility and horizon.
- **Why a proxy, and what it omits.** Free, deep, survivorship-clean **listed-option price history**
  is not available, so the ~10% convex sleeve is approximated by a **rolling 1-year at-the-money SPY
  call marked daily with Black–Scholes** (Black & Scholes, *"The Pricing of Options and Corporate
  Liabilities"*, JPE 1973). The proxy is *transparent* and *deliberately conservative-to-optimistic*:
  it prices the call off **realized** vol (a real call trades at **implied** vol, which is higher — the
  variance risk premium), and it lets the call-holder implicitly keep the underlying's path while a
  real call-holder **forgoes the dividend**. Both tilts *flatter* the strategy; the premium sweep
  restores the variance-risk-premium reality, and both are named on the Signal axis.
- **The variance risk premium — why renting upside is negative-carry.** Options embed an **implied
  volatility above subsequently-realized volatility** (Bakshi & Kapadia 2003; Carr & Wu, *"Variance
  Risk Premiums"*, RFS 2009): the option *seller* is paid for bearing crash risk, so the *buyer* of
  the 10% call sleeve pays that premium every roll. Our `prem_mult` knob scales the Black–Scholes
  price up to that reality (IV/RV ≈ 1.1–1.4), which is what turns the fair-price Sharpe parity into a
  clear shortfall.

## What we measure, and the honesty rails

- **Excess-of-cash, both sides.** Every book is measured net of the BIL bill return. A *constant*
  fraction of SPY funded from cash earns `w·(SPY − cash)` in excess of cash — a pure rescaling — so
  its excess-Sharpe equals SPY's; that identity is *why* the matched-static and buy-and-hold Sharpes
  coincide, and why the honest read on the option is the **convexity spanning alpha**, not a Sharpe
  gap. A de-risked linear book already ties SPY's Sharpe; the question is whether the *non-linearity*
  adds anything (it does not, *t* = +0.33).
- **Leverage/level-clean convexity test.** The spanning alpha regresses 90/10's excess return on the
  matched static-mix excess return (rebalanced to 90/10's realized average delta-weight); the HAC *t*
  on the intercept asks whether the option's spot-dependent convexity earns anything beyond statically
  holding the same average linear exposure.
- **One documented lag, zero look-ahead.** The option position and the roll on day `t` are set from
  the state known at the **close of `t−1`**; today's spot move cannot change today's holdings (a test
  pins this by perturbing only the last day).
- **Robust inference.** Newey–West (HAC) *t* on the daily excess-return difference; a circular
  block-bootstrap CI on the excess-**Sharpe difference**; a two-era cut; a **premium** (variance-risk-
  premium) sweep and a cost sweep; the up/down capture asymmetry.
- **Short history is named on the Signal axis.** BIL lists 2007-05-30, bounding the joint window to
  ~19 years — a single, GFC-and-2020-anchored cycle. Favourable for a protection rule (it contains
  the crashes) but still one draw, not a law.
- **The synthetic control proves only the machinery.** A seeded bear tape (capital must be protected),
  a calm null (premium bleeds, upside given up) and an up-jump tape (the convex floor holds) — never
  cited for the stamp.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* on the excess-return difference and the convexity-alpha intercept).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap behind the
  excess-Sharpe-difference confidence interval.
- **Carr, P. & Wu, L. (2009)** — "Variance Risk Premiums"; the empirical IV-over-RV wedge the premium
  sweep uses to cost the option realistically.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily closes**: SPY (`auto_adjust=True`, total return), BIL (`auto_adjust=True`, total
  return), ^IRX (13-week T-bill rate, in percent), 2007-05-31 → 2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [897-cppi-floor](../../897-cppi-floor/) — **Constant-Proportion Portfolio Insurance**: a dynamic
  floor traded **in the underlying** (multiplier × cushion in SPY, de-risking as the cushion falls),
  *no options*. This study buys the convex payoff **with an actual option premium** (a call sleeve),
  so its cost is an explicit *premium* (the variance risk premium) rather than CPPI's implicit
  cash-drag / whipsaw.
- [617-crash-insurance-cost](../../617-crash-insurance-cost/) — the standing bleed of buying **puts /
  tail hedges** on an already-invested equity book (insuring the *downside*). 90/10 flips the
  structure: it is **out of equity** (in bills) and buys **calls** for the *upside*, so the option is
  a participation instrument, not a hedge overlay.
- [173-four-percent-rule](../../173-four-percent-rule/) — a *withdrawal / decumulation* rule on a
  fixed stock-bond mix (a spending-sustainability question), not a convex option overlay; different
  axis entirely.
- [68-all-weather](../../68-all-weather/) — a *static risk-balanced* multi-asset allocation
  (stocks / bonds / gold / commodities) seeking all-regime robustness by diversification. 90/10 is a
  concentrated **two-leg (bills + SPY-call)** convexity trade, not a diversified risk-parity mix.

None of the siblings tests **T-bills + a rolling long call ("rent the upside, insure the capital")**
priced against its option premium — the Gross/Bodie 90/10 rule — which is this study's own axis.
