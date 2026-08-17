# References & literature map — Study 945 (The Hidden Financing)

## The claim under test

- **The invisible margin loan.** A 2x fund borrows one dollar for every dollar of your NAV;
  a 3x fund borrows two. That borrowing has a price, and the price is nowhere on the fact
  sheet: the prospectus quotes an expense ratio, and the financing cost arrives silently
  inside the swap spread and the futures basis. The claim under test is that the rate is
  *recoverable* from the tape — that a plain regression of the fund's daily return on the
  benchmark's isolates it — and that once recovered it can be raced against what a broker
  would charge you for the same loan.
- **Why it should be recoverable.** The fund's return identity is rigid:
  `r_fund = L·r_index − (L−1)·f/252 − ER/252 + tracking noise`. The slope is *L*, the
  intercept is the whole daily drag. Nothing about that requires a forecast; it is
  accounting, which is why the *t* statistics here are an order of magnitude larger than
  anything on this desk's timing studies.
- **The steelman for the wrapper.** ProShares finances at institutional swap levels on
  billions of notional and lends the collateral back out; a retail margin desk finances one
  account at a rate card. The wrapper *should* borrow far more cheaply than you can — the
  question is how much of that saving it keeps.

## The mechanics of a constant-leverage wrapper

- **Cheng & Madhavan (2009), *The Dynamics of Leveraged and Inverse Exchange-Traded Funds*,
  Journal of Investment Management.** The canonical derivation of the daily-reset identity,
  the end-of-day rebalancing flow (`L·(L−1)·|r|` of NAV) and the path dependence that follows.
  This study uses that identity in the opposite direction: not to predict the NAV, but to
  invert it for the financing term.
- **Avellaneda & Zhang (2010), *Path-Dependence of Leveraged ETF Returns*, SIAM Journal on
  Financial Mathematics.** The closed-form decomposition of a leveraged NAV into the levered
  index return, the variance-drag term and the financing/fee term — the third term is exactly
  what is measured here, and the paper is explicit that it is empirically the least studied.
- **Charupat & Miu (2011), *The Pricing and Performance of Leveraged Exchange-Traded Funds*,
  Journal of Banking & Finance.** Documents tracking performance of the Canadian leveraged
  line and attributes the shortfall between the multiple and the realised return to fees plus
  financing — the same decomposition, on a different tape and an earlier decade.
- **Lu, Wang & Zhang (2012), *Long Term Performance of Leveraged ETFs*, Financial Services
  Review**, and **Trainor & Baryla (2008), *Leveraged ETFs: A Risky Double That Doesn't
  Multiply by Two*, Journal of Financial Planning** — the long-horizon shortfall literature
  that motivates separating drag *shape* (variance) from drag *level* (financing).

## Why the benchmark basis decides the answer

- **The price-index trap.** "The S&P 500" as quoted is a *price* index. A wrapper whose swaps
  reference total return, regressed on the price index, appears to be financed at a **negative**
  rate — our real-tape cross-check returns −1.78% for SSO. The correct benchmark is a
  total-return series, and the residual gap to the *index* total return is the benchmark ETF's
  own fee, added back explicitly. This is the single largest source of wrong published numbers
  on the subject and is why the study prints both.
- **Dividend-yield arithmetic.** The error term is exactly `L ×` the index dividend yield, so a
  3x fund is mis-priced by ~5.5 pp/yr on a ~1.8% yield — larger than the entire true drag.

## Related desk studies (dedup)

- **[Study 61 — Slow-Burn](../../61-slow-burn/)** and **[Study 100 — Melting-Ice](../../100-melting-ice/)**:
  both price the **volatility drag** of the daily reset (`0.5·L·(L−1)·σ²`) and both *assume* an
  all-in fee (Study 100 uses ≈5%/yr for a 3x fund). Study 945 does not assume it — it **measures**
  it (5.07%/yr for UPRO, gratifyingly close) and splits it into expense ratio and interest.
- **[Study 30 — House-Edge](../../30-house-edge/)**: races a levered timing model under two
  honest financing models separated by a *broker mark-up over T-bills*, which it must assume.
  Study 945 supplies that mark-up empirically for the ETF wrapper: **+0.68 pp** on the loan,
  **+1.14 to +1.57 pp** all-in.
- **[Study 943 — Reset Frequency](../../943-leverage-reset-frequency/)** and
  **[Study 944 — Optimal Leverage](../../944-optimal-leverage-realized/)**: the *shape* of the
  reset and the *amount* of leverage the tape rewarded. Both take the cost of leverage as given;
  945 is the study that prices it.
- **[Study 941 — Double-Short Leveraged Pair](../../941-double-short-leveraged-pair/)** and
  **[Study 942 — Inverse ETF Structural Loss](../../942-inverse-etf-structural-loss/)**: the
  short-side wrappers. 945 is long-only and prices the *long* leverage.
- **[Study 593 — HFEA](../../593-hfea-leveraged-6040/)** and
  **[Study 594 — Leverage-Rotation-200SMA](../../594-leverage-rotation-200sma/)**: use UPRO/TQQQ as
  building blocks inside an allocation rule; neither asks what the wrapper charges for the loan.
- **[Study 154 — Leverage-Anomaly](../../154-leverage-anomaly/)** and
  **[Study 260 — Margin-Debt](../../260-margin-debt/)**: corporate leverage as a cross-sectional
  characteristic, and aggregate margin debt as a macro signal — different objects entirely.

## Method lineage

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.ols_hac`](../lev_financing/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_ci`](../lev_financing/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Rolling-window pass-through regressions.** The overlapping-window caveat is standard
  (Hansen & Hodrick, 1980, *Forward Exchange Rates as Optimal Predictors*, JPE): overlapping
  252-day windows leave residuals massively autocorrelated, so the rolling slope is reported as
  descriptive and the inference bar is carried by the full-sample regression.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of slice
  and the content fingerprint printed above every headline table.

## Data sources & labelled assumptions

- **SSO** (ProShares Ultra S&P500, 2x), **UPRO** (ProShares UltraPro S&P500, 3x), **SPY**
  (benchmark), **BIL** (1-3M T-bill ETF, the investable cash leg) — daily **total-return**
  closes via `yfinance` (`auto_adjust=True`). Total return on both sides matters here: the
  wrappers distribute part of their collateral income, and a price-only series would book those
  distributions as extra drag.
- **^IRX** — the 13-week Treasury bill **discount rate**, quoted in percent per annum. It is a
  *rate*, not a wealth index, and is never compounded as one. BIL's realised total return
  (1.267%/yr against ^IRX's 1.396% mean) is used wherever an *investable* cash leg is needed;
  the 0.129 pp gap is BIL's own fee.
- **PROXY — the discount/BEY basis of ^IRX.** A bank-discount quote understates the yield an
  investor earns: 1.396% discount ≈ **1.420% bond-equivalent** over this window
  (`BEY = 365d/(360 − 91d)`). Every "spread over ^IRX" reported is therefore ~2-3 bp too
  generous to the study's own conclusion. It is left unadjusted and labelled rather than
  silently corrected, because the correction runs *against* the finding and is an order of
  magnitude smaller than it.
- **ASSUMPTION — Reg T eligibility in the margin race.** US retail margin accounts are capped
  at 2x *initial* leverage (Federal Reserve Regulation T, 12 CFR 220), so the 3x do-it-yourself
  replication and the sub-1% prime-broker tier are **price comparisons, not choices an ordinary
  account can make**. No margin call is modelled either. Both omissions favour the DIY arm,
  which makes the reported break-even conservative for the wrapper.
- **^GSPC** — the S&P 500 **price** index, used only for the wrong-benchmark cross-check.
- **PROXY / ASSUMPTION — expense ratios.** SSO 0.89%, UPRO 0.91%, SPY 0.0945%, read off the
  prospectuses in 2026. They have drifted a few basis points over the sample. The *drag* is
  measured; only its split between fee and financing depends on these, and the headline is swept
  across ±15 bp (the spread stays positive throughout).
- **PROXY / ASSUMPTION — broker margin spreads.** 0.75% / 1.50% / 4.00% / 6.00% over the
  benchmark rate, indicative of public rate cards (prime-broker tier, low-cost retail, mainstream
  discount broker, full-service retail). Tier- and balance-dependent, and they move; they are
  never used as a single number, only as a swept grid, and the study's actual output is the
  **break-even spread**, which contains no broker assumption at all.
- **Survivorship.** SSO and UPRO are the *survivors* of a leveraged-ETF cohort that has lost many
  members to closure. The estimate is unbiased for these two wrappers; read as a statement about
  the asset class it is a floor, not an average.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps. The
  common window starts at UPRO's June-2009 inception — a *pre-stated* data constraint, not a
  chosen start date. The excluded 2006-2009 stretch of SSO's own history was checked rather
  than assumed away: realised beta there is **1.885** (*t* vs 2 = **−2.85**, i.e. genuinely not
  2, so the intercept would be soaking up a slope error) and it implies a *larger* mark-up
  (+3.60 pp, *t* = 1.56 — noisy). Dropping it therefore **lowers** the headline; it is not a
  window chosen to flatter the result.
