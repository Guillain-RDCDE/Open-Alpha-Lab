# References & literature map — Study 943 (Reset Frequency)

## The claim under test

- **The folk claim.** Leveraged ETFs "decay" because they **reset leverage daily**: every
  evening SSO and UPRO trade back to exactly 2x and 3x, which in a choppy market means
  buying after up-days and selling after down-days. The standard retail conclusion —
  repeated on every forum thread about TQQQ and UPRO — is that the daily reset is the
  villain, and that a **monthly** reset (or "reset only when you have to") would keep the
  leverage without the decay. The claim is mechanical and therefore fully testable: build
  the monthly-reset account explicitly, on margin, financed at the bill rate plus a
  spread, and race it against the funds that actually exist.
- **The steelman.** In a mean-reverting market a *less* frequent reset genuinely helps,
  because you stop rebalancing into every reversal. This is not folklore — it falls out of
  the compounding algebra below. The open questions are how large the effect is on the real
  S&P tape, whether it survives being measured *risk-adjusted*, and what it costs in the
  risks a monthly reset introduces.

## The compounding mathematics

- **Cheng & Madhavan (2009), *The Dynamics of Leveraged and Inverse Exchange-Traded
  Funds*, Journal of Investment Management 7(4).** The canonical derivation: a daily-reset
  fund's multi-period return is a path-dependent function of the underlying, with a drag
  term of roughly −½·L·(L−1)·σ²·T. The source of the "volatility decay" language, and the
  paper that shows the drag is a *consequence of the constant-leverage constraint*, not a
  fee or a defect.
- **Avellaneda & Zhang (2010), *Path-Dependence of Leveraged ETF Returns*, SIAM Journal on
  Financial Mathematics 1(1).** Formalises the same object in continuous time and shows the
  tracking relation is exact in the σ→0 limit — i.e. the gap between a constant-leverage
  and a buy-and-hold-leverage position is *entirely* realised variance along the path. This
  study's efficiency-ratio decomposition is the discrete, monthly-bucket version.
- **Giese (2010), *On the Risk-Return Profile of Leveraged and Inverse Investment
  Products*, Journal of Asset Management 11.** Shows explicitly that constant-leverage
  strategies are advantaged in **trending** markets and disadvantaged in **mean-reverting**
  ones, and that the optimal rebalancing frequency therefore depends on the autocorrelation
  regime. This is the paper our sign result reproduces — and it is why the folklore, which
  states the case unconditionally, is wrong on half the sample.
- **Jarrow (2010), *Understanding the Risk of Leveraged ETFs*, Finance Research Letters
  7(3)** and **Trainor & Baryla (2008), *Leveraged ETFs: A Risky Double That Doesn't
  Multiply by Two*, Journal of Financial Planning 21(5).** The practitioner-facing
  statements of the same arithmetic, including the observation that the drag is second
  order to the *leverage* itself over long horizons — consistent with our finding that a
  monthly reset's extra return is bought with extra average exposure.

## Why the effect exists on this particular tape

- **Lo & MacKinlay (1988), *Stock Market Prices Do Not Follow Random Walks*, Review of
  Financial Studies 1(1)**, and the large literature that followed on the post-2000 sign
  flip in short-horizon index autocorrelation. Daily S&P returns have shown *negative*
  first-order autocorrelation for most of the last two decades, which is exactly the
  condition under which a less frequent reset earns more. Our +1.05 bps/day at 2x is a
  restatement of that stylised fact through a leverage lens — and the synthetic control
  confirms the machinery produces **zero** on an iid tape.
- **Kaufman (1995), *Smarter Trading*.** The efficiency ratio |Σ r| / Σ|r| used here as
  the month's path-shape statistic — a trend/chop measure with no directional content.

## What the monthly reset costs you

- **Federal Reserve Regulation T** (initial margin) and **FINRA Rule 4210** (a 25%
  *maintenance* margin minimum on long equity, with brokers commonly requiring 30–35%).
  These are the source of this study's maintenance-margin PROXY, and the reason a drifting
  3x monthly-reset account is not a thought experiment: on the real tape it is liquidated.
  A daily-reset *fund* is not subject to a margin call at all — its leverage is reset
  inside the fund, and the shareholder's loss is bounded at the share price.

## Related desk studies (dedup)

- **[Study 61 — Slow-Burn](../../61-slow-burn/)** and
  **[Study 100 — Melting-Ice](../../100-melting-ice/)**: do 3x funds "decay to zero"?
  Both study the daily-reset fund *as it is* against its 1x index. Study 943 does the
  thing neither did — **builds the alternative** (a monthly-reset account, financed and
  margined) and races the two reset frequencies against each other.
- **[Study 942 — The Inverse Tax](../../942-inverse-etf-structural-loss/)**,
  **[Study 944 — How Much Leverage](../../944-optimal-leverage-realized/)** and
  **[Study 945 — The Hidden Financing](../../945-leverage-financing-cost/)**: the three
  nearest neighbours on the desk, and each varies a different knob of the same margin
  account. 942 flips the *sign* of the exposure, 944 varies the *multiple* (reset held
  daily throughout), 945 prices the *borrow*. 943 fixes the multiple, the sign and the
  financing and varies only **when the leverage is trued up** — which is why its 25%
  maintenance-margin proxy, irrelevant to a daily-reset account, is the number that
  decides its 3x verdict.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: the rebalancing bonus in a
  multi-asset portfolio. Same family of question (does rebalancing pay?), different object:
  102 rebalances *between assets*, 943 rebalances *leverage on one asset*, where the
  rebalance is a financing decision rather than an allocation one.
- **[Study 836 — Rebalance Timing Luck](../../836-timing-luck/)**: how much of a strategy's
  Sharpe is the arbitrary *day* you rebalance on. 943 asks about the *frequency*, and its
  answer (the frequency choice is a bet on trend versus chop that you cannot forecast) is
  the natural companion to 836's answer about the phase.
- **[Study 593 — HFEA](../../593-hfea-leveraged-6040/)** and
  **[Study 594 — Leverage-Rotation-200SMA](../../594-leverage-rotation-200sma/)**: both are
  *allocation* studies that use UPRO/TQQQ as ingredients — when to hold leverage, and
  against what. 943 holds the allocation fixed and varies only the **mechanics of the
  wrapper**.
- **[Study 590 — Sharpe-Hacking](../../590-sharpe-hacking/)**: leverage and vol-targeting
  cannot manufacture Sharpe. 943 is a live instance — the monthly reset's +17.1% CAGR
  against the daily reset's +15.5% buys **+0.022** of Sharpe, because the extra return is
  the extra exposure.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../reset_freq/strategy.py), [`strategy.ols_hac`](../reset_freq/strategy.py)
  and [`quantlab.analytics`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.sharpe_diff_tstat`](../reset_freq/strategy.py).
- **Sub-period contrasts carry a test of the difference.** The house bar (see
  [METHODOLOGY](../../../METHODOLOGY.md)) forbids reading "+0.047 early, +0.004 late" as a
  decay without testing the gap between them —
  [`strategy.era_difference_test`](../reset_freq/strategy.py) regresses the daily gap on an
  era dummy with a HAC *t*. Here it says the two halves are **not** distinguishable
  (*t* = −1.15 at 2x), so the study does not claim a decay.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_sharpe_ci`](../reset_freq/strategy.py) and
  [`strategy.bootstrap_diff_ci`](../reset_freq/strategy.py) (blocks drawn jointly so the
  paired advantage keeps its day-by-day pairing).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of
  slice and the content fingerprint printed above every headline table.

## Data sources

- **SPY** (the underlying index tape), **SSO** (2x S&P 500, daily reset), **UPRO** (3x S&P
  500, daily reset) and **BIL** (1-3M T-bill, the cash leg) — daily **total-return** closes
  via `yfinance` (`auto_adjust=True`), 2004 → 2026-06-30, cached in the shared desk cache
  `studies/_cache`.
- **^IRX** — the CBOE 13-week Treasury bill **discount yield**, quoted in percent. It is a
  *rate*, not a price, and is never treated as one: it is lagged one day and converted to a
  daily financing rate in `strategy.financing_rate`. Two labelled approximations: the
  discount yield is used directly rather than converted to a bond-equivalent ACT/360 basis
  (a few bps at 5% levels), and the **spread over it is an assumption**, swept 0→200 bps.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  BIL's 2007 inception gates the 2x window and UPRO's June 2009 launch gates the 3x one;
  the 2008 crisis is therefore reached only through the SPY-based stress run, which is
  labelled as such.
