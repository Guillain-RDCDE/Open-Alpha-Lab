# Beat-7 worked complement — "can you slow it down to escape the cost wall?"

*The daily-rebalanced book has a real gross edge (Sharpe 0.70) but a break-even cost of just 3.31 bp,
inside the realistic equity cost band. The standard rescue: hold each day's weights for `h` days to cut
turnover. Does any holding period lift the **net** Sharpe to something investable? Run by
[`examples/verify.py`](../examples/verify.py) on the real S&P 500 panel; the apparatus is exercised on
the synthetic control by [`slingshot/extension.py`](../slingshot/extension.py).*

## The result — a marginal, non-investable rescue

| hold days | 1 | 2 | 5 | 10 | 21 |
|---|---|---|---|---|---|
| gross Sharpe | 0.70 | 0.76 | 0.40 | 0.14 | 0.23 |
| **net Sharpe @5 bp** | −0.36 | −0.00 | −0.11 | −0.14 | **+0.10** |
| turnover/day | 0.63 | 0.45 | 0.29 | 0.14 | 0.07 |

Slowing the book down does exactly what it should to costs — turnover falls ~9× from daily to a 21-day
hold. And unlike Rip-Tide (where the rescue was flatly `BUSTED`), here it *just* crosses into positive
territory: a **net Sharpe of +0.10 at a 21-day hold**. But +0.10 is statistically indistinguishable from
zero over this sample, comes with the survivorship caveat, and is below any reasonable hurdle — it
confirms a *real* premium hiding under the turnover, not an investable strategy. The honest read: the
edge is real, the rescue recovers a sliver of it, and that sliver is not worth trading.

## Why this differs from Rip-Tide — and what the pair proves

On deep futures (Study 32) the same rescue never cleared zero, because there was no gross premium to
begin with. Here the rescue *does* poke above zero, because there genuinely **is** a cross-sectional
reversal premium in single stocks — it is simply too small to survive the turnover it demands. Together
the two studies bracket the effect precisely: short-term reversion is **a single-stock, liquidity-
provision premium**, real but micro-cap-flavoured and cost-dominated, and absent altogether from the
deepest, most-arbitraged instruments.

## Forks worth a PR

- **Liquidity-tiered book** — run the reversal only on the *least* liquid quintile of names (where the
  premium is largest) but model their *higher* spreads honestly; does the net frontier ever win?
- **Sector-neutralisation** — demean within GICS sector instead of the whole index, to strip residual
  industry bets from the signal.
- **Crisis-conditional sizing** — scale the book up only when realised dispersion / VIX is high (Nagel
  2012), the regime where liquidity provision is best paid.
