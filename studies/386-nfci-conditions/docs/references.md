# References & literature map — Study 386 (NFCI-Conditions)

## The claim under test

- **The index (Chicago Fed).** Brave, Scott A., and R. Andrew Butters (2011/2012),
  *"Monitoring financial stability: A financial conditions index approach,"* Federal Reserve
  Bank of Chicago *Economic Perspectives*. The **National Financial Conditions Index (NFCI)** is
  a weekly summary of ~100 measures of risk, credit and leverage across money, debt and equity
  markets, extracted as the first principal factor and standardised so **positive = tighter than
  average, negative = looser**. The **Adjusted NFCI (ANFCI)** isolates the part of conditions
  uncorrelated with the business cycle. Published weekly on FRED as `NFCI` / `ANFCI`.
- **The folklore.** Across macro-Twitter, newsletters and the financial press the NFCI is sold
  as an equity regime switch: *when conditions tighten (NFCI rises through zero / turns
  positive), de-risk — step out of stocks; when they loosen, get back in.* The intuition is that
  tightening credit and rising risk premia choke off the liquidity that fuels equities. We
  steelman exactly this "tight ⇒ sell" timing rule.

## Why the true NFCI is not used here — and what we do instead

- **FRED is unreachable from this environment.** The St. Louis Fed endpoints
  (`fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI`) time out / reset here, so the genuine
  weekly NFCI/ANFCI cannot be pulled. We therefore **construct a transparent financial-conditions
  proxy** entirely from yfinance instruments and z-score it to NFCI's sign convention
  (**high = tight**): equity volatility (`^VIX`), rates volatility (`^MOVE`, the ICE BofA
  bond-option-implied-vol index), a corporate **credit-spread** proxy (investment-grade `LQD`
  total return vs duration-matched Treasuries `IEF`), and **broad-dollar momentum** (`UUP`).
  These are the tradable cousins of NFCI's own risk/credit/leverage sub-indices. It is a
  *narrower, noisier* gauge — and, critically, it is **built partly from equity vol**, so it is
  mechanically glued to stocks; we say so on the Signal axis and rest the verdict on the
  **forward**, not contemporaneous, link.
- **Financial-conditions indices generally.** Hatzius, Hooper, Mishkin, Schoenholtz & Watson
  (2010), *"Financial Conditions Indexes: A Fresh Look after the Financial Crisis,"* NBER WP
  16150 — the canonical survey of how FCIs are built and what they do and don't forecast. The
  Goldman Sachs FCI, Bloomberg US FCI and the Kansas City Fed's KCFSI are close relatives;
  Adrian, Boyarchenko & Giannone (2019), *"Vulnerable Growth"* (American Economic Review), show
  that tight conditions widen the **left tail** of future GDP — an effect on *risk*, much more
  than on the *mean*, which echoes our finding that the proxy cuts equity volatility without
  lifting return.

## Contemporaneous vs predictive — the core methodological trap

- **A coincidence is not a forecast.** Any conditions index that embeds equity volatility (the
  real NFCI does, via the VIX, term-spread and risk-premium blocks) will be strongly *negatively*
  correlated with the *same week's* equity return — that is near-tautological and carries no
  tradable information. The honest test isolates the **forward** link with an execution lag;
  conflating the two is the central error the folklore makes. Cf. the look-ahead / contemporaneous
  pitfalls catalogued in our research-method demos (see *Related desk studies*).
- **Small-sample / regime-concentration inference.** Tight weeks cluster in a handful of
  episodes (2008-09 dominates), so a naive *t* overstates evidence. We test the conditional
  forward mean against the unconditional mean with a **Welch two-sample t** (Welch, 1947) and a
  **placebo / randomization test** (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993), and we put a **stationary / circular block
  bootstrap** (Politis & Romano, 1994) confidence interval on the timing Sharpe gap so the
  volatility-clustering the inference must respect is preserved.
- **Sharpe from vol-cutting is not alpha.** A rule that lowers volatility at unchanged CAGR
  raises Sharpe without adding return; a leverage-matched benchmark neutralises it. The
  distinction between a genuine return edge and a risk-overlay is the desk's Tradability bar.

## Method lineage (the desk's shared engine)

- **Welch t + placebo p-value.** [`strategy.welch_t`](../nfci_conditions/strategy.py) and
  [`strategy.placebo_pvalue`](../nfci_conditions/strategy.py) — conditional (tight-week) vs
  unconditional forward returns, plus a 20,000-draw randomization null sized to the event count.
- **Contemporaneous-vs-forward split.**
  [`strategy.contemporaneous_corr`](../nfci_conditions/strategy.py) measures the mechanical
  same-week link; [`strategy.forward_returns`](../nfci_conditions/strategy.py) measures the
  tradable forward link with a single, documented 1-week execution lag (no look-ahead).
- **Timing backtest with costs.**
  [`strategy.backtest_timing`](../nfci_conditions/strategy.py) — step out when tight, 1-week
  lag, one-way bps per switch, raced excess-of-cash vs buy-and-hold; reports Sharpe, vol and
  CAGR so the vol-vs-return decomposition is explicit.
- **Deterministic synthetic control.**
  [`data.synthetic_fci`](../nfci_conditions/data.py) plants a *contemporaneous* coupling (no
  foresight) and a *separate, knobbed forward edge*; with the edge at zero the forward test must
  stay near t = 0 despite a strong contemporaneous correlation, proving contemporaneous ≠
  predictive. The offline core runs with no network.

## Data sources used here

- **yfinance** weekly closes for `^GSPC`, `^VIX`, `^MOVE`, `LQD`, `IEF`, `UUP`,
  2002-01-04 → 2026-06-19, cached under `_cache/nfci_proxy.csv`. All headline numbers are pinned
  in [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 268 — Sahm-Rule](../../268-sahm-rule/)**: the unemployment-based recession trigger —
  another "macro nowcaster as market-timer" claim, where a real-time recession signal is asked
  to do equity timing it wasn't built for.
- **[Study 118 — Fed-Model](../../118-fed-model/)**: a macro/valuation gauge sold as an
  equity-allocation switch; same genre of "the indicator describes the present, not the future."
- **[Study 317 — Fed-Balance-Sheet](../../317-fed-balance-sheet/)**: liquidity-conditions-as-driver
  of equities, the QE/QT cousin of the financial-conditions thesis.
