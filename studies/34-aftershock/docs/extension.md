# Beat-7 worked complement — "the drift-decay curve: how long does the aftershock last, and does it pay?"

*PEAD is not a one-day pop — it is a slow drift over weeks. The two questions that follow: (1) what does
the **drift-decay curve** look like — does the surprise-signed cumulative abnormal return rise steadily
and then flatten, the Bernard-Thomas (1989) signature? and (2) does the drift persist long enough, and
pay enough, to clear the cost of holding the book? Both are exercised on the synthetic control by
[`aftershock/extension.py`](../aftershock/extension.py) and **measured on the real tape** by
[`examples/verify.py`](../examples/verify.py) (cached EDGAR EPS → SUE × the S&P 500 price panel — see
[`results.md`](results.md) for the fingerprinted run).*

## The drift-decay curve — the shape *is* the anomaly

Line up every earnings event at τ = 0 and average the **surprise-signed** cumulative abnormal return by
trading-day-since-announcement. On the synthetic control (where a real surprise→drift relationship is
baked in):

| days since event | 0 | 10 | 20 | 40 | 60 | 69 |
|---|---|---|---|---|---|---|
| mean signed CAR | +0.0004 | +0.0037 | +0.0059 | +0.0116 | +0.0145 | +0.0153 |

The curve **rises roughly linearly for the first ~40-60 days and then flattens** — the classic PEAD
under-reaction picture (Bernard-Thomas 1989, Fig. 1): the market keeps re-pricing the surprise for weeks,
and then the information is fully absorbed and the drift stops. On the **null** (same surprises, no baked
drift) the curve is flat (**+0.0032** at day 69, vs +0.0153 on the control), confirming the curve measures
information in the surprise, not an artefact of the construction.

The *shape* is the whole point: it tells you the drift is a multi-week phenomenon, so a book must hold for
weeks — which sets the turnover, and therefore the cost.

## The holding-period sweep — does the drift outpay the roll cost?

Hold each name's surprise for `h` trading days, then read gross Sharpe, net Sharpe (@5 bp) and turnover:

| hold days | 5 | 20 | 40 | 60 | 90 |
|---|---|---|---|---|---|
| gross Sharpe | 1.10 | 2.78 | 3.62 | 3.52 | 3.50 |
| **net Sharpe @5 bp** | 0.10 | 2.26 | **3.24** | 3.22 | 3.28 |
| turnover/day | 0.562 | 0.166 | 0.087 | 0.057 | 0.042 |

The story matches the decay curve exactly. **Too short (5 days)** and you capture only the first sliver of
the drift while paying the most turnover — the net Sharpe collapses to ~0.10. **Around 40-60 days** the
book has banked most of the cumulative drift at low turnover, and the net Sharpe peaks (**+3.24**).
**Beyond ~60 days** the drift has flattened, so extending the hold neither adds nor costs much — the
frontier plateaus. The sweet spot is "hold for about a quarter," precisely the window over which the
drift-decay curve was still rising.

## On the real tape — the drift is there, and slow, and too small to pay

Run on real EDGAR SUEs × the S&P 500 price panel (see [`results.md`](results.md)), the same two diagnostics
tell the honest story:

- **The drift-decay curve `CONFIRMED`** on real earnings: the surprise-signed CAR rises monotonically from
  **+0.0014 (day 0) to +0.0115 (day 69)** — Bernard-Thomas's shape, on the tape, not just the control.
- **The holding-period sweep** shows you must hold the full quarter: net @5 bp is **−0.66 at a 5-day hold**
  (all turnover, no drift) and only reaches break-even **(+0.05) at a 60-day hold** — the drift is genuine
  but so slow and small that even the optimal horizon barely clears 5 bp.

On the synthetic control the holding-period frontier clears costs comfortably (break-even ~57 bp) because
the baked drift is clean. **The real tape is far harsher, and predictably so:** real PEAD is *small* and on
a liquid large-cap universe its **break-even cost is only 6.0 bp** — *inside* the realistic equity
round-trip band, and that is before correcting the **survivorship bias** that inflates it. The drift is
real; harvesting it net is a `MIRAGE`.

## Forks worth a PR

- **Liquidity / size tiering** — run the book within liquidity quintiles and model each tier's *honest*
  spread; quantify how much of PEAD's edge lives only in the un-tradable small names.
- **SUE bucketing** — sort into SUE deciles (Bernard-Thomas style) and trade only the extreme tails,
  where the drift is strongest, against the cost of a more concentrated book.
- **Earnings-momentum vs price-momentum** — overlay the price-momentum signal (Chordia-Shivakumar 2006)
  and ask how much incremental drift the *earnings* surprise adds once price momentum is controlled.
