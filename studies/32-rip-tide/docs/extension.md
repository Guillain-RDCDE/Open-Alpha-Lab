# Beat-7 worked complement — "can you save reversion by slowing it down?"

*The base contrarian book trades every day and dies on costs (break-even 0.79 bp). The standard rescue
for a too-fast signal is to **rebalance less often**: hold each position for `h` days instead of one,
which cuts turnover — and therefore cost — roughly in proportion to `h`. The question this complement
pre-registers: does **any** holding period lift the net Sharpe above zero? Run by
[`examples/verify.py`](../examples/verify.py) on the real 18-futures tape (as-of 2026-06-10, fingerprint
`b8a35a878ebc`); the machinery is exercised on the synthetic control by
[`rip_tide/extension.py`](../rip_tide/extension.py). Gross and net are labelled in every row.*

## The result — `BUSTED`

| hold days | 1 | 2 | 5 | 10 | 21 |
|---|---|---|---|---|---|
| gross Sharpe | +0.25 | +0.07 | +0.02 | −0.12 | −0.14 |
| **net Sharpe @2 bp** | −0.39 | **−0.30** | −0.19 | −0.22 | −0.20 |
| turnover/day | 1.41 | 0.83 | 0.49 | 0.25 | 0.13 |

Slowing the book down works *exactly* as intended on the cost side — turnover falls from 1.41/day at
daily rebalancing to 0.13/day at a 21-day hold, a >10× cut. But the net Sharpe **never crosses zero**,
because the gross edge fades even *faster* than the cost: +0.25 at a 1-day hold is already down to +0.07
at two days and negative past ten. The reason is structural, not a tuning failure: whatever bounce these
futures have is exhausted within a day or two, so a *slower* version of the same edge does not exist —
and that is the only thing a longer hold can harvest.

## Why the synthetic behaves differently — and what that proves

On the synthetic Ornstein-Uhlenbeck control (`revert_strength = 0.06`), the *same* holding-period sweep
keeps a **net Sharpe above 1.8 even at a 21-day hold**, because there the overshoot genuinely persists
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
