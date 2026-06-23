# References & literature map — Study 382 (Treasury cash-futures Basis Trade)

## The claim under test

- **The trade.** The Treasury **cash-futures basis trade**: buy a deliverable cash Treasury
  (financed in the repo market) and sell the corresponding Treasury **future**, capturing the
  small gap (the *basis*) as it converges to zero at delivery. Because the position is
  duration-hedged and the gap is tiny, desks run it at **extreme leverage** (often 50–100×
  via repo) to turn a few basis points of *net carry* (coupon income minus repo cost) into a
  double-digit return on equity. The pitch is "near-riskless free carry."
- **Who makes it.** It is a flagship **relative-value** trade for fixed-income hedge funds.
  The mechanics (implied repo rate, cheapest-to-deliver, net basis) are textbook: Burghardt,
  Belton, Lane & Papa, *The Treasury Bond Basis* (3rd ed., 2005) — the canonical practitioner
  reference. Galen Burghardt & Terry Belton, *The Treasury Bond Basis* underpins the
  implied-repo framing we model here.
- **Why regulators worry.** The trade's *size* and *leverage* make it a financial-stability
  concern. The **Federal Reserve** (FEDS Notes, e.g. *"Hedge Fund Treasury Trading and Funding
  Fragility,"* 2023) and the **Bank for International Settlements** (BIS Quarterly Review,
  *"The market turmoil and the basis trade,"* 2020; and follow-ups on hedge-fund Treasury
  positioning) document how the basis trade amplified the **March 2020** dash-for-cash: a repo
  funding shock forced levered unwinds, dislocating the world's most liquid bond market until
  the Fed intervened. The **Office of Financial Research** and **FSB** echo the concern.

## Why true basis data isn't free — and what we do instead

- **Cash-vs-futures basis & GC repo.** The clean inputs — deliverable cash-bond prices,
  Treasury futures with the cheapest-to-deliver, and general-collateral repo rates — live on
  paid terminals (Bloomberg, terminals' `DLV`/basis pages), not on the free yfinance endpoint
  (per-ticker OHLCV only). We therefore build a **transparent carry / implied-repo MODEL**,
  labelled as such throughout: net carry = the **10-year yield** (`^TNX`) minus a **short
  funding rate** (`^IRX`, the 13-week T-bill as a **repo proxy**), with the residual
  mark-to-market proxied by a small-duration (0.5y) move in that spread. Every input is a
  public series; the model is a methodological choice, not a fabrication.
- **The repo proxy.** The 13-week T-bill yield tracks GC repo closely in normal times; the
  two *diverge* precisely in funding stress (repo spikes above bills), which means our model
  **understates** the funding-shock tail — conservative in the direction that matters.

## Why "free carry" is the wrong frame — the statistics

- **Sharpe is leverage-invariant.** Scaling a return stream by a constant scales mean and
  volatility identically, leaving the Sharpe ratio unchanged (Sharpe, 1966/1994, *The Sharpe
  Ratio*). A levered carry trade's eye-popping return-on-equity is therefore *not* evidence of
  a better risk-adjusted edge — it is the same edge multiplied. The honest Signal test is the
  **HAC *t* of the unlevered mean** (Newey & West, 1987, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica).
- **Carry = short volatility.** Systematic carry strategies earn a smooth premium punctuated
  by rare crashes — negatively skewed, fat-tailed returns. Koijen, Moskowitz, Pedersen & Vrugt
  (2018), *Carry* (Journal of Financial Economics), document the carry premium across asset
  classes and its crash risk; Brunnermeier, Nagel & Pedersen (2008), *Carry Trades and Currency
  Crashes* (NBER Macro Annual), formalise the "picking up pennies in front of a steamroller"
  tail. Our excess-kurtosis ≈ 10 is exactly this signature.
- **Leverage, funding liquidity, and margin spirals.** Brunnermeier & Pedersen (2009),
  *Market Liquidity and Funding Liquidity* (Review of Financial Studies), explain the
  loss → margin-call → forced-deleveraging → larger-loss spiral that turns a small basis
  dislocation into a systemic event under high leverage — the March-2020 mechanism we stress.

## Method lineage (the desk's shared engine)

- **HAC inference + Sharpe.** [`strategy.hac_t`](../treasury_basis_trade/strategy.py) and
  [`strategy.sharpe`](../treasury_basis_trade/strategy.py) — the Signal-axis tests on the
  *unlevered* (leverage-invariant) return.
- **Sign-randomisation placebo.**
  [`strategy.placebo_pvalue`](../treasury_basis_trade/strategy.py) — flip the carry's sign at
  random; the share of randomised series whose Sharpe beats the real one (the honest
  "could autocorrelated noise fake this?" null).
- **Leverage / tail / funding-shock.**
  [`strategy.leverage_table`](../treasury_basis_trade/strategy.py),
  [`strategy.tail_stats`](../treasury_basis_trade/strategy.py) and
  [`strategy.funding_shock`](../treasury_basis_trade/strategy.py) — the Tradability axis: the
  same Sharpe levered to 50–100×, its drawdown/CVaR, and a March-2020 stress unwind.
- **Deterministic synthetic control.**
  [`data.synthetic_basis`](../treasury_basis_trade/data.py) plants a known unlevered Sharpe;
  the offline core runs with no network. The control confirms the inference is faithful *and*
  that leverage-invariance + an exploding drawdown is mechanical.
- **Execution lag.** The carry is earned with a **1-day lag** (signal at the close of *t*
  earns *t+1*) — one shift, applied once, in [`data.basis_model`](../treasury_basis_trade/data.py).

## Data sources used here

- **yfinance** daily closes for `^TNX`, `^IRX`, `IEF`, 2002-07-31 → 2026-06-18, cached under
  `_cache/basis_inputs.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 364 — FX-Carry-Trade](../364-fx-carry-trade/)**: the currency cousin — the same
  "earn the rate differential" carry, the same negative-skew crash tail. Reading the two
  together shows carry's free-lunch reputation is the same illusion across asset classes.
- **[Study 132 — Yield-Curve-Steepener](../132-yield-curve-steepener/)**: another rates
  relative-value trade built on the same term-structure spread the basis trade's carry comes
  from.
