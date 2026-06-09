# Sources & literature map — Study 12 (Paper-Prophet)

## The claim under test

- **Roan (@RohOnChain), *How To Build A Time Series Model To Win Every Single Trade (Quant
  Framework)*** (X / Twitter article, 19 May 2026; ~792 K views; the author bills himself a
  "backend developer working on system design, HFT-style execution, and quantitative trading
  systems" focused on prediction markets). The post prescribes a complete time-series stack on SPY:
  (1) test stationarity with Augmented Dickey–Fuller and model **returns, not prices**; (2) fit
  **ARIMA(1,0,1)** on a rolling 252-day window for a one-step directional forecast; (3) fit
  **GARCH(1,1)** on the residuals for one-step volatility; (4) size `min(1, 1/σ̂)` and trade
  `sign(forecast) × size`, walk-forward; (5) report Sharpe and max drawdown. This study ports the
  `TimeSeriesTradingSystem` class verbatim ([`paper_prophet/stack.py`](../paper_prophet/stack.py))
  and decomposes its Sharpe into the **forecast** and the **vol-targeting** terms.

- **The author's own concession.** Part 5 of the article states *"the GARCH-based position sizing
  is doing more work than the ARIMA forecast direction"* and quotes directional accuracy at *"52 to
  55 percent."* This is not a strawman we impose; it is the article's own claim, and it is the
  hypothesis the study operationalises — that the stack is vol-targeting with a forecast bolted on.

- **Companion thread.** The same framework family includes the Markov-chain piece tested in
  **[Study 10 — Markov-Mint](../../10-markov-mint/)** (a different author, same "win every single
  trade" genre). The article closes by asking when GARCH and a Markov regime model *disagree* — an
  open thread we flag in beat 7.

## What the article gets right (kept, not disputed)

- **Unit roots & spurious regression.** **Dickey & Fuller (1979)** (ADF test), **Granger & Newbold
  (1974)** (*Spurious Regressions in Econometrics*). The article's core teaching lesson — never
  regress price-on-price (R² ≈ 0.99 is a unit-root artefact), difference to returns first — is
  correct and we reproduce the ADF check as a sanity step, not a finding.

- **Walk-forward, not in-sample.** The article rightly warns that fitting on the full series and
  grading in-sample is "a curve fit with a label." We honour this: strict rolling 252-day windows,
  forecast graded one step ahead, no peeking. The study additionally *quantifies* the in-sample
  inflation as a teaching figure (beat 7).

## Why the forecasting half is expected to fail (the null)

- **Fama (1970), *Efficient Capital Markets*** and **Samuelson (1965), *Proof That Properly
  Anticipated Prices Fluctuate Randomly*.** Daily returns on a deep, heavily-arbitraged index carry
  negligible exploitable serial dependence; the conditional mean `E[r_{t+1} | ℱ_t]` is ≈ 0 relative
  to its noise, so an ARIMA(1,0,1) sign is ≈ a coin flip. This is the sharp null for the **Signal**
  axis.

- **ARMA/ARIMA & GARCH machinery.** **Box & Jenkins (1970)** (ARIMA), **Engle (1982)** (ARCH),
  **Bollerslev (1986)** (GARCH). The study does not dispute the machinery — GARCH genuinely models
  volatility clustering, a real stylized fact — only the claim that ARIMA on index returns
  manufactures a tradable directional signal.

## Why the sizing half "works" — but is beta, not alpha

- **Moreira & Muir (2017), *Volatility-Managed Portfolios* (Journal of Finance).** Scaling exposure
  by `1/σ̂` raises the Sharpe of a long equity position because realized volatility is persistent
  and inversely related to forward risk-adjusted returns. This is exactly the GARCH
  position-sizing in the stack — a documented risk-premium harvest available without any forecast.
  It is the mechanism behind the expected **Tradability `MIRAGE`** verdict: the Sharpe is managed
  beta, reproducible by constant-long with the same sizing.

- **Leverage effect.** **Black (1976)**, **Christie (1982)** — the negative vol–return relation that
  powers vol-targeting in equity indices and is weaker/inverted in other assets, motivating the
  other-instruments sweep in beat 7.

- **Kelly / fractional sizing context.** **Kelly (1956)**, **Thorp (2006)** — sizing by inverse risk
  is sound *given* an edge; applied to a coin-flip forecast it adds nothing but the vol-targeting
  tailwind, which is what the decomposition isolates.

## Desk method

- **Newey & West (1987)** (HAC *t*) and the bootstrap — the inference backbone, applied here to the
  realized directional edge per day and to the Sharpe difference (stack vs vol-targeting control).
  Shared engine: [`../../quantlab/`](../../quantlab/). House method:
  [`../../METHODOLOGY.md`](../../METHODOLOGY.md).

## Related studies in this repo

- **[Study 10 — Markov-Mint](../../10-markov-mint/)** — the companion "win every single trade"
  pipeline on a prediction market; same genre, same null-free framework, verdict NONE / MIRAGE.
- **[Study 08 — True-Strength](../../08-true-strength/)** — another "better indicator" sold as new
  that reduces to the same trade repainted; same direction-vs-substance question.
- **[Study 04 — Social-Oracle](../../04-social-oracle/)** — a viral retail signal sold on a
  framework rather than a null; same beat-3 "announce the falsification first" discipline.
