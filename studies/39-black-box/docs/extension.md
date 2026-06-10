# Beat-7 worked complement — the shuffled-label control (the overfitting smoking gun)

*The MLP posts a gorgeous **in-sample** Sharpe and accuracy. Is it learning structure, or memorising
noise? The cleanest test in the López de Prado / Bailey-Borwein-López de Prado-Zhu playbook: destroy any
real relationship between features and target by **randomly shuffling the labels**, then re-fit and
re-score in-sample. If the in-sample fit survives shuffled — and therefore meaningless — targets, it was
never measuring predictive skill. Run by [`examples/verify.py`](../examples/verify.py) on the real crypto
tape; the apparatus is exercised on the synthetic null by
[`black_box/extension.py`](../black_box/extension.py).*

## The result — the in-sample fit is memorisation

In-sample **training accuracy** (how well the net reproduces the labels it was trained on), on BTC-USD:

| labels | in-sample train accuracy |
|---|---|
| **true** | 0.664 |
| shuffled #1 | 0.662 |
| shuffled #2 | 0.668 |
| shuffled #3 | 0.687 |
| shuffled #4 | 0.668 |

The coin-flip baseline is **0.50**. The net reproduces *randomly shuffled* labels in-sample at
**0.66–0.69** — essentially identical to the true labels (**0.664**). Since the shuffled targets carry
**zero** information by construction, this can only be the net memorising whatever it is handed. The
entire in-sample "edge" is fitting capacity, not forecasting skill.

## Why this matches the walk-forward collapse

The shuffled-label control and the in-sample-vs-walk-forward gap are two views of the same fact. The net
has enough free parameters to drive in-sample training accuracy well above 0.5 on *any* label set — true
or shuffled. So a high in-sample accuracy tells you nothing about whether real structure exists. Only the
**walk-forward** number — fit on the past, predict the genuinely unseen future — can, and it sits at a
coin-flip (≈ 0.5) on real daily crypto direction, negative after costs.

## Why this differs from Crystal-Ball — and what the pair proves

[Study 22 (Crystal-Ball)](../../22-crystal-ball/) was a backtest trap of the **first** kind: a look-ahead
leak in the *feature* (a two-sided filter that secretly encoded the future). Black-Box is the **second**
kind: an over-parameterised learner memorising the *labels* in-sample. Both produce a beautiful backtest
out of nothing; both vanish the moment you compute the only number a live trader could have earned.
Together they bracket the two classic ways a machine-learning backtest lies.

## Forks worth a PR

- **Purged & embargoed cross-validation** (López de Prado 2018) — honest model selection that respects the
  serial correlation and label horizon, instead of naïve fit-and-predict.
- **Deflated / probabilistic Sharpe ratio** — correct the in-sample headline for the number of
  architectures, feature banks and lookbacks tried; here it would push an already-zero edge below zero.
- **Richer feature banks / other coins / intraday bars** — does *any* honest, cost-surviving walk-forward
  edge ever appear? Report the walk-forward number, never the in-sample one.
