# References & literature map — Study 351 (BTC 5-minute Polymarket momentum)

## The claim under test

- **The viral pitch.** A widely-shared social post: *"I put in $300, this bot turned it into
  $14,000, and I didn't even code it — I grabbed an open-source repo and let it run."* The
  strategy: on Polymarket **BTC Up/Down 5-minute** markets, enter ~2 minutes before close when
  BTC has already moved $70–100, buy the favoured side (quoted $0.80–0.99) "because the result
  is basically decided," optionally micro-hedging at extreme skew. The testable claims: (H₁) a
  fresh intra-window move predicts the close; (H₂) the favoured side is buyable below its true
  odds; (H₃) the compounding turns small stakes into a fortune.
- **The advertised code.** A public GitHub repo packaged as an "OpenClaw skill" (config
  profiles, watcher scripts, a CLOB runner). Crucially, the published code **delegates order
  placement to an unpublished engine** (`pm_live_trade_runner.py`) that reads a wallet private
  key from a `.env` — i.e. the part that touches funds is not auditable. The security hazard is
  named in the teardown and is, in our view, the dominant risk.

## Market mechanics

- **Polymarket BTC Up/Down 5-minute markets.** Event slug `btc-updown-5m-<unix_start>`;
  resolves **Up** iff the BTC price at the window close is ≥ the price at the open, per the
  **Chainlink BTC/USD** data stream (`data.chain.link/streams/btc-usd`). Shares are binary,
  paying $1 / $0, traded on the Polymarket **CLOB** (central limit order book). Best ask via
  `clob.polymarket.com/price?token_id=…&side=sell`.
- **Binary-share expected value.** A share bought at price *p* that wins with probability *w*
  earns `w(1−p) − (1−w)p = w − p` per share (`(w−p)/p` per dollar staked). The break-even is
  `p = w`; the entire tradability question is whether the quoted price is below the true
  win-rate. This identity is the spine of the teardown.

## Why the edge is priced out — the relevant finance

- **Prediction-market efficiency.** Wolfers & Zitzewitz (2004), *Prediction Markets* (Journal
  of Economic Perspectives) — market prices are close-to-unbiased probability estimates. On a
  liquid, public BTC feed, the favoured side converges to its true odds, driving `p → w` and the
  edge to ~0.
- **The favourite-longshot bias.** Ottaviani & Sørensen (2008), *The Favorite-Longshot Bias: An
  Overview*; Thaler & Ziemba (1988), *Anomalies: Parimutuel Betting Markets*. Short-odds
  favourites tend to be *over*-priced (p > w) and longshots under-priced — i.e. the bias points
  **against** a buy-the-favourite strategy, making `w − p` if anything negative.
- **Endogenous pricing / adverse selection.** Glosten & Milgrom (1985), *Bid, Ask and
  Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. A cheap ask
  on a near-decided market is information, not a gift: it exists where the outcome is genuinely
  uncertain or where the seller is better-informed. The price is endogenous to the signal, so
  measuring `w` alone cannot establish an edge — you must measure `p` at the signal.

## Why high win-rate ≠ profit — sizing and ruin

- **The Kelly criterion and over-betting.** Kelly (1956), *A New Interpretation of Information
  Rate*; Thorp (2006), *The Kelly Capital Growth Investment Criterion*. The Kelly fraction at
  `(w, p)` near fair value is ~0; betting 50% of stack is gross over-betting, which maximises the
  probability of ruin even when the per-bet edge is zero or positive.
- **Gambler's ruin / martingale survivorship.** Feller (1968), *An Introduction to Probability
  Theory*, Vol. 1 (gambler's ruin). A high win-rate with a large adverse tail is the classic
  martingale: many small wins, rare catastrophic loss, a heavy-tailed terminal-wealth law whose
  *median* is ruin and whose rare survivor tells the story. Our Monte-Carlo quantifies it.
- **Brownian-motion continuation (the closed form).** For a driftless arithmetic Brownian price,
  once it leads by *D* with *k* minutes left the side holds with probability `Φ(D/(σ√k))`; the
  measured continuation is the conditional expectation `E[Φ(|m|/σ√k) | |m| ≥ D]`. Used as the
  synthetic positive control, confirming the engine is unbiased and that the *fair* price gives
  EV = 0 exactly.

## Method lineage (the desk's shared engine)

- **Binomial / large-sample inference.** The Signal axis here is predictive accuracy, tested as
  a binomial proportion against 0.5 (`strategy.continuation` returns the *z*). Standard errors
  `√(w(1−w)/n)` on every win-rate bar.
- **Deterministic synthetic control.** A fixed-seed arithmetic-Brownian window generator
  ([`data.synthetic_windows`](../btc5m_polymarket/data.py)) with a closed-form benchmark
  ([`strategy.expected_continuation`](../btc5m_polymarket/strategy.py)) — the offline core runs
  with no network.
- **Read-only live capture.** A paper-trade scaffold that polls public Polymarket/Binance
  endpoints for the favoured ask vs realised outcome — **no key, no order, no money** — feeding
  [`data.load_live`](../btc5m_polymarket/data.py).

## Data sources used here

- **Binance** public klines (BTCUSDT 1-minute), 45 days to 2026-06-20, cached under `_cache/`.
- **Polymarket** Gamma + CLOB public endpoints for the live ask capture; **Chainlink BTC/USD**
  is the markets' resolution source (we proxy the close with the Binance bar; sub-dollar, rare
  discrepancies). All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: the equity-market twin — a real viral
  "90% win-rate" that is the *exit's shape*, not an edge. Same high-win-rate /
  negative-expectancy martingale signature, different venue.
- **[Study 156 — Martingale](../../156-martingale/)** and **[Study 157 — Kelly-Sizing](../../157-kelly-sizing/)**:
  the sizing machinery — why doubling-down and over-betting convert a fair (or losing) game into
  near-certain ruin.
