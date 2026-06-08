# References — Twin-Spread 👯

*Sources for [Study 05](../README.md). The literature map matters here because the effect
under test — **pairs trading / relative-value convergence** — is the most thoroughly
documented anomaly in the book, *and* the most thoroughly documented to have **decayed**.
The honest verdict turns on (a) telling a real historical edge from one that still survives
to a tradeable present, and (b) being explicit that a liquid ~170-name basket is not the
CRSP cross-section the original was formed from.*

## The claim under test

- **The viral thread** — [@MatiasScalbi, x.com/MatiasScalbi/status/2063042609816252666](https://x.com/MatiasScalbi/status/2063042609816252666)
  — restates the GGR (1999) results at full strength: ~1.44%/mo on the top-20 pairs,
  ~0.90% net of bid-ask, Sharpe 0.59 vs 0.09 for the market, near-zero beta, and ~10.4%/yr
  in the 1999–2003 *post-publication* window — the basis for "it kept paying after everyone
  knew." See [the claim](../README.md#1--the-claim).

> The strongest steelman: a **parameter-free** rule (no fitted knobs to overfit) earned a
> market-neutral carry that outlived its own publication. The null: minimum-distance pairs
> are not genuinely cointegrated, so "convergence" is a coin flip whose negative skew
> (pairs that break and never return) drags the mean to zero or below — and decimalisation
> plus crowding finished the job after 2002.

## The strategy, precisely

- **Gatev, Goetzmann & Rouwenhorst (1999/2006), "Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule", *Review of Financial Studies* 19(3).** The canonical
  reference and the exact rule we implement: 12-month formation by **minimum sum of squared
  deviations** of normalized prices, 6-month trading, open at **2σ** divergence, close on
  the price crossing, returns on **committed** vs **employed** capital, and the **one-day
  waiting** robustness for bid-ask bounce. Their headline: ~11%/yr, Sharpe ~0.6, low beta,
  1962–2002 CRSP universe.

## Data

- **Universe** — the cached, deep-history liquid US names (the parquets Study 04 already
  cached). A *liquid basket*, **not** CRSP: ~170 names today, growing from ~7 in the early
  1960s. The study reports the eligible-name count and selected-pair tightness per window,
  because **pair quality is a function of universe breadth** — see
  [`examples/verify_real.py`](../examples/verify_real.py) and [`docs/results.md`](results.md).
- **Prices** — daily close, **split-only** mode (the cached one). Dividends are not folded
  in; the consequence is named and shown to work *against* the rule — see the data-mode
  note in [`pairs_trading/data.py`](../pairs_trading/data.py). Volume rides along for the
  capacity beat.
- **Benchmark** — the equal-weight cross-section return, a local stand-in tape for the
  market-neutrality regression (a dollar-neutral book should show β≈0).

## Literature map — why the edge was real, and why it decayed

- **The decay, documented.** *Do & Faff (2010), "Does Simple Pairs Trading Still Work?",
  Financial Analysts Journal* — profitability declines sharply after 2002; the share of
  pairs that diverge and **never reconverge** rises. *Do & Faff (2012)* adds trading costs
  and finds the net edge largely gone. This is the direct empirical anchor for our
  `Decay → CONFIRMED`.
- **The microstructure cause.** Decimalisation (2001) narrowed quoted spreads and removed
  much of the **bid-ask bounce** that inflated early measured returns; the rise of stat-arb
  desks crowded the convergence. *Bowen, Hutchinson & O'Sullivan (2010)* on intraday pairs
  and cost sensitivity.
- **The richer modern toolkit (what we deliberately did *not* use).** *Avellaneda & Lee
  (2010), "Statistical Arbitrage in the US Equities Market", Quantitative Finance* (PCA /
  mean-reversion of residuals); *Engle & Granger (1987)* and *Johansen (1991)* cointegration
  — the **economic-anchor** filters our naive minimum-SSD rule omits, and the obvious
  beat-7 forks. Testing whether any of these rescues the rule is explicitly *out of scope*
  here: we test the textbook rule the thread sells.
- **Why convergence is short gamma.** The negative-skew structure (many small wins, rare
  large losses) is general to mean-reversion bets; *Duarte, Longstaff & Yu (2007), "Risk and
  Return in Fixed-Income Arbitrage", RFS* makes the "picking up nickels" point that explains
  a >50% win rate with a ≤0 mean.
- **Liquidity & impact.** Square-root market impact and capacity. *Almgren et al. (2005);
  Tóth et al. (2011), "Anomalous price impact and the critical nature of liquidity".*

## Method cross-links

- Bootstrap Sharpe CI and the cost/capacity machinery are the shared desk protocol,
  [`quantlab/`](../../../quantlab/) and the [methodology](../../../METHODOLOGY.md).
- The punchline rhymes with the rest of the desk: like [Study 02](../../02-falling-knife/)
  (the −3% dip) and [Study 04](../../04-social-oracle/) (the guru), a famous, eye-obvious
  pattern turns out to be **no edge once measured honestly** — and like [Study 03](../../03-fear-gauge/),
  the real money, if any, is conditional on a regime (here: dislocations), not a standing rule.
