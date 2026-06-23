# References & literature map — Study 395 (Quantum-Computing-Basket)

## The claim under test

- **The pitch.** Through 2024-2026 a "quantum-computing basket" became a viral retail and
  newsletter trade: hold the handful of **pure-play** quantum stocks — **IONQ** (trapped-ion),
  **Rigetti / RGTI** (superconducting), **D-Wave Quantum / QBTS** (annealing) and **Quantum
  Computing Inc. / QUBT** (photonic) — and "ride the next computing revolution" instead of the
  boring index. The names rocketed (and crashed) on milestone headlines (Google's *Willow* chip,
  Dec 2024; IBM and IonQ roadmaps; periodic "quantum advantage" claims), and the basket's trailing
  return looked spectacular — the classic *thematic-basket* hook.
- **The folklore.** "Quantum is the next AI / the next internet" — a real, transformative
  technology, therefore (the leap) its listed pure-plays must be a buy. The "next big thing" framing
  is precisely what invites a teardown: a real technology is not the same object as a tradable,
  risk-rewarded equity basket, and a pre-revenue cohort that is *up* a lot is exactly where
  volatility, drawdown and survivorship masquerade as edge.

## Why this is a hype-cycle / risk question, not a stock-picking one

- **Hype cycles.** Gartner's *Hype Cycle* framing (Jackie Fenn, 1995; Fenn & Raskino, *Mastering
  the Hype Cycle*, 2008) — the "peak of inflated expectations" and "trough of disillusionment" — is
  the folk model the claim sits inside. Quantum computing has spent years near the peak; the
  question is whether the *equity* basket pays you for the technology or just exposes you to the
  cycle's amplitude.
- **Lottery stocks & the low-vol / idiosyncratic-vol puzzle.** Bali, Cakici & Whitelaw (2011),
  *Maxing out: Stocks as lotteries* (JFE) and Ang, Hodrick, Xing & Zhang (2006), *The cross-section
  of volatility and expected returns* (JF) document that high-idiosyncratic-vol, lottery-like stocks
  earn **lower** risk-adjusted returns on average — the opposite of what a "buy the volatile theme"
  retail trade assumes. The quantum pure-plays are a textbook lottery cohort (triple-digit vol,
  −90%-class drawdowns).
- **De-SPAC underperformance & survivorship.** Most of these names reached the tape via SPAC
  mergers. Klausner, Ohlrogge & Ruan (2022), *A sober look at SPACs* (Yale J. on Regulation), and
  Gahng, Ritter & Zhang (2023), *SPACs* (Review of Financial Studies), document systematic post-merger
  underperformance and high attrition — so a basket of the de-SPAC names that *survived* to a current
  quote is **survivorship-tilted upward** (the failures are absent). Named on the Signal axis.

## Why a huge CAGR need not be an edge — the statistics

- **Sharpe, not return.** Sharpe (1966, 1994), *The Sharpe Ratio*. A +59%/yr CAGR at 160% vol is a
  *lower* Sharpe than a +14%/yr index at 16% vol; the right lens for "is this better?" is
  risk-adjusted, and on that lens the basket loses to both SPY and the diversified ETF.
- **Autocorrelation-robust inference on the spread.** Newey & West (1987), *A simple, positive
  semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix* (Econometrica)
  — the HAC standard error behind the spread's *t*. With one short, single-regime window dominated by
  a few violent up-months, the standard error of a thematic spread is large; the desk's bar is a
  HAC *t* ≥ 2 on the real tape (literature support alone reads `WEAK`).
- **Selection / multiple testing on a famous theme.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (RFS), and Bailey & López de Prado (2014), *The Deflated Sharpe
  Ratio* — a basket discovered ex-post in the data it won in needs a far higher bar than a naive
  point estimate; our synthetic null makes the same point on data where the truth is known.

## Method lineage (the desk's shared engine)

- **HAC t of the spread.** [`strategy.hac_tstat_diff`](../quantum_computing_basket/strategy.py) —
  the Signal-axis test: basket-minus-benchmark monthly spread under a Newey-West long-run variance,
  vs SPY, QQQ and the diversified ETF.
- **Risk-adjusted race.** [`strategy.summarize`](../quantum_computing_basket/strategy.py) and
  [`strategy.race`](../quantum_computing_basket/strategy.py) — CAGR / Sharpe / vol / max-drawdown for
  each leg (the hype-cycle signature) and the per-name decomposition.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../quantum_computing_basket/data.py) plants a *risk-adjusted* edge
  (`edge_sharpe`) on a fat-tailed hype basket; at `edge_sharpe=0` the basket has zero expected
  excess and the HAC *t* stays at the null, so the harness can only certify a planted edge — proving
  the engine is faithful and that fat-tailed beta cannot manufacture significance.
- **Costs with one execution convention.**
  [`strategy.basket_returns`](../quantum_computing_basket/strategy.py) charges one-way turnover × NAV
  per rebalance; the fixed equal-weight basket needs no signal lag (membership is constant).

## Data sources used here

- **yfinance** daily auto-adjusted (total-return-proxy) closes for IONQ, RGTI, QBTS, QUBT, the QTUM
  ETF, SPY and QQQ, resampled to monthly, 2021-05 → 2026-05 (the window over which all four
  pure-plays trade), cached under `_cache/quantum_panel.parquet` (fingerprint `d955f566bc4b`). All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 393 — AI-Datacenter-Basket](../393-ai-datacenter-basket/)**: the closest sibling — a
  thematic basket whose spread is real on the tape but selection-after-the-fact. There the critique
  is ex-post name selection; here it is **risk and sample size** (the spread isn't even significant).
- **[Study 334 — ARK-Innovation](../334-ark-innovation/)**: the disruptive-tech thematic fund — does
  riding "innovation" pay risk-adjusted? Same family of high-vol theme bets.
- **[Study 302 — Lithium-Boom](../302-lithium-boom/)** and
  **[Study 303 — Uranium-Revival](../303-uranium-revival/)**: thematic commodity-equity baskets —
  real theme, the question is whether the *basket* harvests it net of risk.
