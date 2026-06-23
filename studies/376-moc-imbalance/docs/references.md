# References & literature map — Study 376 (MOC-Imbalance)

## The claim under test

- **The closing-auction imbalance trade.** Practitioner and trading-desk lore holds that
  large **market-on-close (MOC) order imbalances** — the net buy/sell pressure submitted to
  the official NYSE/Nasdaq closing auction — **push the last print** away from fair value,
  and that this dislocation **reverses overnight** (the next session's open leans back). The
  fade is a staple of short-horizon equity trading: stand against the imbalance into the
  close, capture the snap-back at the open.
- **Why index-rebalance days are the headline case.** On S&P / Russell reconstitution dates
  and quarterly triple-witching, passive and index funds *must* trade at the official close to
  minimise tracking error, so MOC imbalances are largest there. The believers' strongest
  version of the claim is therefore "the reversal is biggest on rebalance days." We test that
  subset explicitly (third Friday of quarter-end months).

## Why true auction-imbalance data is not free — and what we do instead

- **The MOC imbalance feed is a paid product.** NYSE's *Order Imbalances* and Nasdaq's *Net
  Order Imbalance Indicator (NOII)* are proprietary, fee-liable data feeds; they are **not**
  available through the free yfinance endpoint, which serves daily OHLCV only. We therefore
  **construct a transparent proxy**: the signed intraday **displacement**
  `(close − open)/(high − low)`, a [−1, 1] gauge of how hard price was pushed toward the close
  inside the day's range. This is a coarse, noisy stand-in — it conflates the full session
  path with final-print pressure — and we say so on the Signal axis. Every input is a public
  daily bar.
- **Auction microstructure.** Pagano & Schwartz (2003), *A closing call's impact on market
  quality at Euronext Paris* (Journal of Financial Economics); Bogousslavsky & Muravyev
  (2023), *Who trades at the close? Implications for price discovery and liquidity* — the
  growth of closing-auction volume and its information content. Barclay, Hendershott &
  Jefferis on end-of-day price formation.

## Why the close→open reversal is mostly microstructure

- **Bid-ask bounce inflates apparent overnight reversal.** Roll (1984),
  *A simple implicit measure of the effective bid-ask spread* — when the close prints near a
  quote and the open prints near the other side, the close→open return mechanically *looks*
  like reversal even with no information. The proxy can only **over**-state the reversal, so a
  near-zero measured effect bounds the true one below it.
- **Short-horizon reversal / liquidity provision.** Nagel (2012),
  *Evidence-based expectations: the returns to liquidity provision* (Review of Financial
  Studies) and Lehmann (1990), *Fads, martingales, and market efficiency* — short-horizon
  return reversal is compensation to liquidity suppliers and is **eaten by the spread** for
  anyone who crosses it. Khan, Khurshid & co. on overnight-vs-intraday return decomposition;
  Lou, Polk & Skouras (2019), *A tug of war: overnight versus intraday expected returns* (JFE)
  — the overnight component has its own systematic behaviour distinct from intraday.

## Why a tiny, insignificant effect is not an edge — the statistics

- **HAC inference / the t ≥ 2 bar.** Newey & West (1987),
  *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent
  covariance matrix* (Econometrica) — overnight gaps carry serial correlation and
  heteroskedasticity, so the reversal slope is judged by a **HAC (Newey-West) t-stat**, the
  desk's Signal-axis bar.
- **Randomization / placebo.** Fisher's randomization logic and Efron & Tibshirani,
  *An Introduction to the Bootstrap* (1993) — shuffle the displacement↔gap pairing and ask how
  often chance produces a slope at least as negative. The honest small-effect test when the raw
  R² is ~0.0001.
- **Selection on a famous trade / data-snooping.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies); White (2000),
  *A reality check for data snooping* — a much-repeated practitioner trade needs a higher bar
  than a naive in-sample t-stat, especially once costs are included.

## Method lineage (the desk's shared engine)

- **HAC t + placebo p-value.** [`strategy.newey_west_t`](../moc_imbalance/strategy.py) and
  [`strategy.placebo_pvalue`](../moc_imbalance/strategy.py) — the Signal-axis tests:
  Newey-West slope of the overnight gap on the displacement proxy, and a 20,000-draw
  randomization null.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../moc_imbalance/data.py) plants a *known* overnight-reversal knob
  so the engine can be validated: edge = 0 must NOT manufacture significance, a large planted
  reversal must light up. The offline core runs with no network.
- **Fade strategy with structural lag + costs.**
  [`strategy.fade_returns`](../moc_imbalance/strategy.py) enters today's close and exits the
  next open (a 1-day lag by construction, no look-ahead); costs applied in
  [`strategy.net_of_costs`](../moc_imbalance/strategy.py).

## Data sources used here

- **yfinance** daily raw OHLC for SPY, QQQ, IWM, 2005-01-03 → 2026-06-17, cached under
  `_cache/ohlc.csv`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 01 — Overnight-Anomaly](../01-overnight-anomaly/)**: the broader question of where
  equity returns are earned — overnight vs intraday. The MOC-reversal trade is one slice of
  that decomposition, viewed through the closing auction.
- **[Study 140 — Amihud-Illiquidity](../140-amihud-illiquidity/)**: the same microstructure
  family — an effect that is "real" in the data but lives inside trading frictions, so the
  measured number is paid away by anyone who must cross the spread.
