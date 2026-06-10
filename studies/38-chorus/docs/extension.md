# Beat-7 worked complement — "breadth is the lever, and the blend scheme barely matters"

*The base run shows the diversification mechanism is real but `MIXED` on the real tape. Two follow-ups
sharpen the lesson: (1) how does the combo's Sharpe scale with the **number** of components — the
Fundamental Law's √breadth — and (2) does the clever **inverse-vol (risk-parity)** blend actually beat the
naive **equal-weight** one? Run by [`examples/verify.py`](../examples/verify.py) on the real S&P 500 panel;
the apparatus is exercised on the synthetic control by
[`chorus/extension.py`](../chorus/extension.py).*

## The breadth sweep — √breadth, but only over *good* bets

| # signals | components | gross Sharpe |
|---|---|---|
| 1 | momentum | +0.36 |
| 2 | momentum, reversal | **+0.66** |
| 3 | momentum, reversal, low-vol | +0.00 |

Adding the second (decorrelated, positive-IC) signal lifts the Sharpe ~1.8× — close to the √2 ≈ 1.41 the
Fundamental Law predicts for two independent bets, and a touch above it because momentum and reversal here
are *negatively* nudged in a couple of regimes. Adding the third signal *collapses* it, because that
voice (low-vol) has negative IC on this sample: breadth multiplies IC, and 1/3 of a negative IC is still
negative. **On the synthetic control, where all three carry real edge, the same sweep rises monotonically
1.49 → 2.57 → 2.71** — the apparatus captures the √breadth lift cleanly when the components deserve it.
The real-tape `MIXED` verdict is therefore not a failure of the method; it is the Law's fine print made
visible.

## Equal-weight vs risk-parity — the scheme is not the lever

| scheme | gross Sharpe | net @1 bp |
|---|---|---|
| equal-weight | +0.00 | −0.12 |
| risk-parity (inverse-vol) | −0.15 | −0.31 |

Inverse-vol weighting is supposed to stop a noisy signal dominating. Here it makes things slightly
*worse*: the off-key low-vol leg is the lowest-turnover, middling-volatility signal, so a pure inverse-vol
rule actually *over*-allocates to it relative to its (poor) information content. The clever blend cannot
fix a bad roster — **component selection dominates combination scheme**. The right lever is *which voices
sing*, not *how loud each one is*.

## Why this differs from the synthetic — and what the pair proves

The synthetic control is the alpha-combo's best case: three weak signals, each with genuine independent
edge, blend to a Sharpe above every soloist (2.71 vs ≤2.1) at near-zero pairwise correlation. The real
S&P 500 is the honest case: the components *are* decorrelated (−0.03), the *mechanism* fires for the right
pair (momentum+reversal 0.66 > both parts), but the chorus is only as good as its worst sincere voice, and
its turnover (0.38/day, break-even 0.02 bp) erases it net. Together they bracket the §3.20 claim exactly:
**diversification across decorrelated alphas is real and is the closest thing to a free lunch — but it is
addition of expected returns, not alchemy; a decorrelated loser still loses.**

## Forks worth a PR

- **Sign-screened breadth** — only admit a component whose *trailing* (out-of-sample) Sharpe is positive,
  so a regime-broken signal (here low-vol) is benched automatically; does the screened chorus hold up?
- **Wider roster** — add carry, quality, seasonality (Studies 18/19/28) and re-run the breadth sweep:
  does the √breadth curve keep rising, or do the new voices just add correlated noise?
- **Optimised combine** — replace equal/inverse-vol with a (shrunk) mean-variance blend of the component
  return streams, à la §3.18; does estimating the alpha covariance beat the naive weights enough to clear
  costs, or does estimation error eat the gain (Clarke-de Silva-Thorley)?
