# References & literature map — Study 897 (CPPI Floor)

## The claim under test

- **The source rule.** Fischer **Black & Robert Jones**, *"Simplifying Portfolio Insurance"*
  (Journal of Portfolio Management, 1987) introduce **Constant-Proportion Portfolio Insurance**:
  keep the risky-asset exposure equal to a fixed **multiplier** `m` times the **cushion**
  `C = V − F`, where `F` is a protected **floor**. As the portfolio falls the cushion shrinks and
  the rule automatically de-risks; if the cushion reaches zero the book is fully in the safe asset
  ("cash-locked") sitting on the floor. It is a *self-financing* dynamic strategy — no options
  required — that replicates a protective-put-like payoff by trading the underlying.
- **The definitive comparison.** André **Perold & William Sharpe**, *"Dynamic Strategies for Asset
  Allocation"* (Financial Analysts Journal, 1988) place CPPI against **buy-and-hold** and
  **constant-mix (rebalancing)** strategies and draw the key result we lean on: CPPI is a **convex,
  momentum** strategy (it buys after gains, sells after losses), constant-mix is **concave,
  contrarian**. CPPI *outperforms* in trending markets and *underperforms* in oscillating ones;
  crucially, **it buys its downside protection by giving up upside** — there is no free lunch, only
  a reshaping of the return distribution.
- **The floor that does not ratch up.** Textbook CPPI's floor accretes only at the cash rate, so
  after a long run-up it no longer sits under the elevated NAV and the *drawdown-from-peak*
  protection fades — the motivation for **Time-Invariant Portfolio Protection (TIPP)** (Estep &
  Kritzman, *"TIPP: Insurance Without Complexity"*, Journal of Portfolio Management, 1988), which
  ratchets the floor up to a fixed fraction of the running high-water mark. We test the classic
  cash-accreting floor and *name* the fading-peak-protection effect in the era cut.
- **Gap risk — the un-hedged tail.** Because the book rebalances at discrete (daily) prices, a
  single overnight jump can carry the portfolio through the floor before the rule can react. At full
  exposure `w = m·C/V` a one-day drop `d` breaches the floor **iff `d > 1/m`** — the multiplier is
  literally the reciprocal of the crash the floor can absorb. The CPPI "gap risk" / "cash-lock"
  literature (e.g. Cont & Tankov, *"Constant Proportion Portfolio Insurance in the Presence of
  Jumps in Asset Prices"*, Mathematical Finance, 2009) formalises this; we stress it directly.

## What we measure, and the honesty rails

- **Excess-of-cash, both sides.** Every book is measured net of the BIL bill return. A *constant*
  fraction of SPY funded from cash earns `w·(SPY − cash)` in excess of cash — a pure rescaling — so
  its excess-Sharpe equals SPY's; that identity is *why* the matched-static and buy-and-hold Sharpes
  coincide, and why the honest read on CPPI's dynamics is the **spanning alpha**, not a Sharpe gap.
- **Leverage/level-clean re-timing test.** The Perold–Sharpe / Moreira–Muir **spanning alpha**
  regresses CPPI's excess return on the matched static mix's excess return; the HAC *t* on the
  intercept asks whether the *dynamic re-timing* earns anything beyond statically holding the same
  average risk. A raw return difference would confound the average-weight level with the timing.
- **One documented lag, zero look-ahead.** The CPPI exposure on day `t` is computed from the cushion
  known at the **close of `t−1`**; today's return cannot move today's weight (a test pins this).
- **Robust inference.** Newey–West (HAC) *t* on the daily excess-return difference; a circular
  block-bootstrap CI on the excess-**Sharpe difference** (so the volatility clustering survives the
  resample); a two-era cut; multiplier / floor / cost sweeps; and an explicit gap-risk stress.
- **Short history is named on the Signal axis.** BIL lists 2007-05-30, bounding the joint window to
  ~19 years — a single, GFC-and-2020-anchored cycle. Favourable for an insurance rule (it contains
  the crashes) but still one draw, not a law.
- **The synthetic control proves only the machinery.** A seeded bear tape (the floor must hold), a
  calm null (nothing to protect) and a gap knob (breach iff drop > 1/m) — never cited for the stamp.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* on the excess-return difference and the spanning-alpha intercept).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap behind the
  excess-Sharpe-difference confidence interval.
- **Moreira, A. & Muir, T. (2017)** — "Volatility-Managed Portfolios"; the spanning-regression
  framing used to make the re-timing test leverage/level-clean.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, **total return**): SPY, IEF, BIL, 2007-05-31 →
  2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [617-crash-insurance-cost](../../617-crash-insurance-cost/) — the standing cost of **buying puts**
  / tail hedges outright (an *options* premium bleed). This study buys a put-*like* payoff
  **synthetically by trading the underlying** (CPPI), with no option premium but a *cash-drag* cost
  and un-hedged gap risk instead.
- [624-buffer-etf-cost](../../624-buffer-etf-cost/) — packaged **defined-outcome / buffer ETFs**
  (an option-collar structure sold in a wrapper, with a cap). CPPI is a *self-managed dynamic*
  overlay with no cap and no wrapper fee — a different mechanism for the same "limit the downside"
  wish.
- [659-costless-collar](../../659-costless-collar/) — a static **options collar** (long put funded
  by a short call) that finances protection by selling upside *once*. CPPI finances protection
  **dynamically**, by mechanically de-risking as the cushion falls, not by selling a call.
- [30-house-edge](../../30-house-edge/) — the desk's reference teardown on why a mechanical betting/
  sizing rule is not an edge; CPPI is the portfolio-insurance instance of the same lesson — a
  distribution reshaping, priced fairly, not free alpha.

None of the siblings tests the **cushion-multiplier dynamic floor traded in the underlying** — the
Black–Jones / Perold–Sharpe CPPI rule — which is this study's own axis.
