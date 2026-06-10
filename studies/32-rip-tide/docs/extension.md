# Beat-7 worked complement — "can you save reversion by slowing it down?"

*The base contrarian book trades every day and dies on costs (break-even 0.24 bp). The standard rescue
for a too-fast signal is to **rebalance less often**: hold each position for `h` days instead of one,
which cuts turnover — and therefore cost — roughly in proportion to `h`. The question this complement
pre-registers: does **any** holding period lift the net Sharpe above zero? Run by
[`examples/verify.py`](../examples/verify.py) on the real 18-futures tape; the machinery is exercised on
the synthetic control by [`rip_tide/extension.py`](../rip_tide/extension.py).*

## The result — `BUSTED`

| hold days | 1 | 2 | 5 | 10 | 21 |
|---|---|---|---|---|---|
| gross Sharpe | 0.08 | 0.20 | 0.03 | −0.05 | −0.15 |
| **net Sharpe @2 bp** | −0.56 | **−0.17** | −0.18 | −0.16 | −0.20 |
| turnover/day | 1.38 | 0.82 | 0.48 | 0.25 | 0.13 |

Slowing the book down works *exactly* as intended on the cost side — turnover falls from 1.38/day at
daily rebalancing to 0.13/day at a 21-day hold, a >10× cut. But the net Sharpe **never crosses zero**.
The reason is structural, not a tuning failure: the gross edge fades as fast as the turnover does,
because on these deep futures there was no durable, slow-moving overshoot to harvest. You can only save a
fast signal by slowing it down when a *slower* version of the same edge exists; here it does not.

## Why the synthetic behaves differently — and what that proves

On the synthetic Ornstein-Uhlenbeck control (`revert_strength = 0.06`), the *same* holding-period sweep
keeps a **net Sharpe above 1.7 even at a 21-day hold**, because there the overshoot genuinely persists
for many days (half-life ≈ `ln 2 / 0.06` ≈ 12 days). That is the point of the control: it shows the
rescue *would* work if a slow reversion existed — so its failure on the real tape is evidence about the
*market*, not about the method. Deep futures simply do not overshoot in a way you can fade after costs.

## Forks worth a PR

- **Cross-sectional reversion** — fade the *relative* winners within an asset class instead of each
  market vs its own past; intra-sector reversal can outlive time-series reversal.
- **Illiquid universe** — re-run the identical book on a basket of *single-stock* futures or small-cap
  ETFs, where the liquidity-provision premium (Avramov–Chordia–Goyal 2006) should reappear.
- **Crisis-conditional sizing** — Nagel (2012) shows reversal pays most when liquidity evaporates; gate
  the book on a stress signal (VIX, realised vol) and trade it only when the premium is rich.
