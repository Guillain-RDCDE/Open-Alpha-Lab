# References & literature map — Study 156 (Martingale)

## The claim under test

- **The folk recipe.** A fixture of retail trading forums, Reddit, and YouTube: *"When your
  stock drops 5%, buy more — you're lowering your average cost. When it drops another 5%,
  double up again. When it recovers, you profit on all those shares. It's a mathematical system
  that turns losing trades into winning ones."* The claim is essentially that averaging-down
  creates a systematic edge over buy-and-hold by exploiting the convexity of the average-entry
  price. We steelman it as: *the martingale (doubling) strategy on a single instrument produces
  a positive expected return in excess of a buy-and-hold of the same total capital, with an
  acceptable tail-risk profile.* The recipe is, by its own framing, a bet that temporary
  adverse price moves are mean-reverting and that finite capital is sufficient to see them through.

## Why the claim has surface appeal — the real math it invokes

- **Jensen's inequality / average-cost lowering.** Doubling after price falls genuinely lowers
  the average entry price. On a path that goes down then recovers to the initial level, the
  martingale *does* profit while buy-and-hold breaks even. The psychological appeal is real.
  The flaw is the conditioning: a path that recovers is a survivor; a path that keeps falling
  triggers ruin. The expected value averages across all paths, not the nice-looking ones.
- **Dollar-cost averaging (DCA).** Statman (1995), *A Behavioral Framework for Dollar-Cost
  Averaging* (Journal of Portfolio Management), argues DCA reduces regret but not expected cost.
  The martingale is an exponentially-amplified version: it helps on recovering paths and destroys
  on declining ones, with the downside path's impact growing exponentially while the upside is
  capped at the take-profit.

## The mathematics of ruin — why it can't work with finite capital

- **Gambler's ruin theorem.** Feller, W. (1950), *An Introduction to Probability Theory and Its
  Applications*, Vol. 1 (Wiley). The fundamental result: on a fair random walk with a fixed
  amount of capital and an absorbing barrier (ruin), the probability of ruin is a function of
  the capital fraction and the bet size. With finite capital and an opponent with infinite
  capital (the market), ruin probability → 1 as time → ∞. The martingale is a direct
  instantiation: it delays ruin probabilistically at the cost of exponential capital growth.
- **Doubling strategy and no-arbitrage.** In a no-arbitrage market (formal statement: Harrison
  & Pliska 1981, *Martingales and Stochastic Integrals in the Theory of Continuous Trading*,
  Stochastic Processes and Their Applications), the martingale betting strategy cannot create
  an expected profit exceeding the risk-free rate when capital is constrained. The continuous-time
  analogue of our discrete episode test confirms the theoretical null.
- **Fat tails amplify ruin.** Mandelbrot (1963), *The Variation of Certain Speculative Prices*
  (Journal of Business), and the subsequent literature on power-law tails of equity returns
  (Gabaix et al. 2003, *A Theory of Power-Law Distributions in Financial and Social Sciences*,
  QJE) imply that the size of adverse moves can exceed any step threshold instantaneously
  (gap openings, circuit breakers, earnings surprises), making the discrete step-down model
  optimistic — in practice ruin can occur in a single session.

## The win-rate / negative-skew illusion

- **Skew and expectation.** Taleb (2004), *Fooled by Randomness* (Random House), and Taleb
  (2007), *The Black Swan* (Random House), provide the canonical popular treatment of
  strategies that manufacture a high win-rate at the cost of rare catastrophic losses ("picking
  up pennies in front of a steamroller"). The martingale is the textbook example: ~94% of
  episodes end in a small win; ~6% end in total ruin.
- **Win-rate is not expectancy.** The study's sister, [Study 72 — Loaded-Dice](../../72-loaded-dice/),
  demonstrates the same illusion via the fixed-tick trap on 5-minute bars: a small take-profit
  with a far stop manufactures a 90%+ win-rate whose expectancy is still ≈ 0. The martingale
  creates the same structure via a different mechanism (capital doubling vs exit asymmetry).
- **Behavioral persistence.** Shefrin & Statman (1985), *The Disposition Effect: Explaining
  Why Investors Hold Losers and Sell Winners* (Journal of Finance), document that retail
  investors tend to average down as a natural disposition effect. The martingale formalises
  this behaviour into a rule, but formalisation does not create edge.

## Inference and method

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../martingale/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Ornstein-Uhlenbeck mean reversion.** Uhlenbeck & Ornstein (1930), *On the Theory of
  Brownian Motion* (Physical Review) — the synthetic positive-control tape uses a
  discrete-time OU process to plant controlled mean-reversion at the step-down scale.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), SPY from 1993, QQQ from 1999, AAPL and
  GE from 1993, all through 2026-06-12. The long daily history (~30 years) provides adequate
  power (~592 non-overlapping episodes pooled). The offline reproducible core and test-suite
  run on [`data.synthetic_daily`](../martingale/data.py), never the network.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the direct sibling — same negative-skew
  illusion produced by exit asymmetry on a 5-minute crossover scalp.
- **[Study 30 — House-Edge](../../30-house-edge/)**: the "house always wins" cost-drag study.
- **[Study 33 — Slingshot](../../33-slingshot/)**: short-term reversal (the only scenario where
  mean-reversion at the step scale is real, and averaging-down would legitimately help).
- **[Study 101 — Slow-and-Steady](../../101-slow-and-steady/)**: position sizing and the
  Kelly criterion — the rational capital allocation that maximises long-run growth.
