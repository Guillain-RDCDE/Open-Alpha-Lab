# References & literature map — Study 14 (Gamma-Gospel)

## The claim under test

- **The GEX pitch.** *"Most retail traders are looking at the wrong thing… the actual input is
  dealer hedging."* — GEX Edge (@GEXEdgeIO), *"Gamma Exposure (GEX) Explained"*, June 2026,
  <https://gexedge.io>. The steelman: net dealer **gamma exposure**, computed before the open from
  the options chain, sets the *character* of the day. **Positive gamma** ⇒ dealers long gamma ⇒
  they sell rips and buy dips ⇒ a **range** day with suppressed vol; **negative gamma** ⇒ dealers
  short gamma ⇒ they chase the move ⇒ a **trend** day with amplified vol. *"The regime question —
  are dealers suppressing or amplifying — is usually more important [than direction]."* The post
  layers on the gamma flip, call/put walls, and the 0DTE/charm/vanna machinery, and sells a daily
  regime read at $7/month. We test the load-bearing core: **does the sign of GEX forecast the
  day's realised character — and does it do so *beyond* the volatility regime (VIX)?**

- **The convention the whole map rests on.** GEX is not observable; it is *assumed*. The standard
  retail / SqueezeMetrics construction takes customers to buy index puts and overwrite calls, so
  dealers are **long call gamma and short put gamma**: `GEX = Σ_calls Γ·OI·100·S² − Σ_puts
  Γ·OI·100·S²`. That dealer-positioning assumption is this study's single load-bearing modelling
  choice — flagged as such in the README and in [`gamma_gospel/data.py`](../gamma_gospel/data.py).

## Why the steelman is *almost* right — the real effect underneath

- **Dealer gamma and realised volatility.** There is genuine academic support that option dealers'
  hedging dampens or amplifies volatility. Barbon & Buraschi (2020), *Gamma Fragility* (working
  paper), and the practitioner literature (Nomura's Charlie McElligott; SqueezeMetrics, *The
  Implied Order Book*) document that aggregate dealer gamma correlates with subsequent realised
  vol and intraday mean-reversion. The honest question is **how much of that is just the volatility
  regime**: high-VIX environments are mechanically both more volatile *and* (being put-heavy)
  negative-gamma, so a raw "negative gamma → more vol" gap is half-tautological. This study isolates
  the increment over VIX.
- **Pinning at high open-interest strikes.** Ni, Pearson & Poteshman (2005), *Stock Price
  Clustering on Option Expiration Dates* (Journal of Financial Economics), document price pinning to
  large-OI strikes near expiry — the kernel of truth under the "call wall / put wall / gamma flip"
  levels. We note it as a going-further test rather than the headline.
- **Charm / vanna and the afternoon drift.** The pitch's advanced section (charm-driven afternoon
  flows, vanna under IV compression) maps to documented end-of-day and OPEX-window patterns, but
  these are second-order to the regime claim and are left to beat 7.

## Method lineage (the desk's shared engine)

- **Autocorrelation-robust inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  Bartlett-kernel HAC errors used on every coefficient here, because daily outcomes conditioned on a
  persistent regime are autocorrelated. Implemented in
  [`decompose.hac_ols`](../gamma_gospel/decompose.py); the engine's
  [`quantlab/analytics.py`](../../../quantlab/analytics.py) carries the same machinery.
- **The confound / partialling-out move.** The verdict rests on a nested regression `y ~ vix` vs
  `y ~ vix + neg_gamma` ([`decompose.partial_over_vix`](../gamma_gospel/decompose.py)): the
  surviving coefficient and its incremental R² are what distinguish a real effect from a relabeled
  one. This is the same test Study 12 used to strip a forecast back to vol-targeting.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and content fingerprint every headline run carries (live options/VIX data drifts and extends).

## Data sources used here

- **Alpha Vantage `HISTORICAL_OPTIONS`** (<https://www.alphavantage.co/documentation/>). Full
  historical SPY option chains by `(symbol, date)` since 2008-01-01, with per-contract `gamma`,
  `open_interest`, `implied_volatility` and the other greeks — one of the few sources carrying
  **both open interest and gamma historically**, but a **premium** endpoint (the free key is
  rejected with *"this is a premium endpoint"*). The free options sources we checked — DoltHub's
  `post-no-preference/options` and OptionsDX — carry greeks but **no open interest**, so they cannot
  weight a GEX. Net: a reliable historical GEX needs a **paid** chain source, which is why Study 14
  ships pre-registered. One request per trading-day chain; `fetch_chain` caps each run out loud
  (house rule: no silent caps).
- **Yahoo! Finance** (via `yfinance`): daily SPY OHLC (the realised character: Parkinson range vol
  and directional efficiency) and `^VIX` close (the confound), pinned with `as_of` and
  fingerprinted; and — via `snapshot_chain` — the *live* SPY option chain as a **free, key-less**
  GEX source, with the caveat that it is a snapshot (no history) and its open interest is often
  sparse/unreliable, so it serves a single-day read or a forward-accumulated panel rather than a
  powered backtest.

## Related desk studies

- **Study 12 — Paper-Prophet**: the same "is it just the volatility regime?" teardown — there an
  ARIMA+GARCH "forecast" reduced to vol-targeting; here a dealer-gamma "regime" tested against VIX.
  *Vol-targeting in a trenchcoat* is the recurring desk verdict.
- **Study 03 — Fear-Gauge** and **Study 06 — Clockwork-Vol**: the other VIX-centric studies —
  whether the fear gauge is tradable, and whether it runs on a timeable cycle.
