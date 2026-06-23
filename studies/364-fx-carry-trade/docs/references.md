# References & literature map — Study 364 (FX Carry Trade)

## The claim under test

- **The trade.** Borrow a low-interest-rate currency (the *funding* currency — classically
  JPY or CHF) and invest the proceeds in a high-interest-rate one (the *investment*
  currency — AUD, NZD, ...). If spot exchange rates were a fair bet you would lose the
  rate differential back through depreciation; empirically you do not, for long stretches.
  The realised excess return — the *carry* — has historically been positive with a high
  Sharpe, which is why the carry trade is the canonical "macro free lunch."
- **Uncovered interest parity (UIP) and its failure.** UIP says
  $\mathbb{E}[\Delta s_{t+1}] = i_t - i_t^*$: the high-yield currency should depreciate by
  exactly the interest-rate gap, so carry should earn zero in expectation. The empirical
  rejection of UIP is one of the oldest puzzles in international finance — Fama (1984),
  *Forward and spot exchange rates* (Journal of Monetary Economics); Hansen & Hodrick
  (1980). The forward premium predicts the *wrong* sign of the spot change at short
  horizons, leaving a positive carry.

## Why carry is not a free lunch — the crash skew

- **Carry returns are negatively skewed.** Brunnermeier, Nagel & Pedersen (2009),
  *Carry Trades and Currency Crashes* (NBER Macroeconomics Annual): carry trades earn a
  premium in calm regimes and suffer sudden, severe losses in risk-off episodes
  ("currency crashes"), so the unconditional Sharpe overstates the reward — the left tail
  is fat. High-interest currencies have negative conditional skewness; the carry is
  compensation for crash risk, not arbitrage.
- **Carry as a risk premium.** Lustig, Roussanov & Verdelhan (2011), *Common Risk Factors
  in Currency Markets* (Review of Financial Studies) build the **HML-FX** carry factor and
  show it loads on a common global risk factor — carry pays you for bearing systematic FX
  risk, which is exactly why it crashes when that risk materialises. Menkhoff, Sarno,
  Schmeling & Schrimpf (2012), *Carry Trades and Global Foreign Exchange Volatility*
  (Journal of Finance): carry returns are strongly negatively related to *global FX
  volatility* innovations — a volatility-risk premium in disguise.
- **The short-volatility shape.** The carry payoff resembles writing out-of-the-money
  options / selling insurance (Jurek (2014), *Crash-neutral currency carry trades*,
  Journal of Financial Economics): hedging the crash tail with options removes much of the
  premium, evidence that the premium *is* the crash compensation. The desk has documented
  the same sold-insurance shape in equity short-vol carry.

## Why a high Sharpe with negative skew is not "real edge" here

- **Skewness and the inadequacy of mean/Sharpe.** A Sharpe ratio summarises the first two
  moments only; a sold-insurance strategy can post a high Sharpe for years while its true
  risk lives in the third and fourth moments. We therefore report **skewness**, the
  **worst month**, **max drawdown**, and a **crash-conditional split** alongside the mean,
  and judge the mean with a **Newey-West / HAC t-stat** (Newey & West, 1987, *A simple,
  positive semi-definite, heteroskedasticity and autocorrelation consistent covariance
  matrix*, Econometrica) rather than an i.i.d. t that ignores the volatility clustering.
- **Selection and post-publication decay.** The carry premium has been heavily mined and
  has weakened in the post-2008 zero-rate era as rate differentials compressed
  (McLean & Pontiff, 2016, *Does Academic Research Destroy Stock Return Predictability?*,
  Journal of Finance, applied to factors generally). A flat-rate G10 world earns little
  carry while still carrying the crash tail — the worst of both.

## Method lineage (the desk's shared engine)

- **HAC t-stat + Sharpe + skew.** [`strategy.hac_t`](../fx_carry_trade/strategy.py),
  [`strategy.annualised_sharpe`](../fx_carry_trade/strategy.py),
  [`strategy.skewness`](../fx_carry_trade/strategy.py) — the Signal-axis tests on the
  monthly basket return.
- **Crash accounting.** [`strategy.crash_split`](../fx_carry_trade/strategy.py) and
  [`strategy.max_drawdown`](../fx_carry_trade/strategy.py) quantify how much of the premium
  is given back in the worst-decile months.
- **Deterministic synthetic control.**
  [`data.synthetic_fx`](../fx_carry_trade/data.py) plants a known carry premium *and* a
  known fat-tailed crash; the offline core runs with no network. With the carry spread set
  to zero the control must NOT manufacture significance; with a planted edge it must light
  up; the planted crash skew must surface as deep negative skewness.
- **Costs with borrow.** [`strategy.net_basket`](../fx_carry_trade/strategy.py) charges a
  one-way trading cost on the gross book plus a borrow spread on the short leg
  (shorts pay borrow — house rule).

## Data sources used here

- **yfinance** daily FX spot for a fixed G10 basket vs USD (`EURUSD=X`, `GBPUSD=X`,
  `AUDUSD=X`, `NZDUSD=X`, `USDJPY=X`, `USDCHF=X`, `USDCAD=X`, `USDNOK=X`, `USDSEK=X`),
  normalised to USD-per-foreign units, 2004-01-01 → 2026-06-19, cached under
  `_cache/fx_spot.csv`. Carry is a **fixed per-currency proxy** (a long-run average
  annualised short-rate differential vs USD) — FRED deposit rates are not bundled here, so
  the carry is an explicit, transparent constant, named a proxy throughout. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 147 — FX-Momentum](../147-fx-momentum/)**: the *other* canonical FX factor on
  the same G10 universe — ranking by trailing return rather than by carry. Read together
  they map the two classic currency premia and how both fared post-publication.
