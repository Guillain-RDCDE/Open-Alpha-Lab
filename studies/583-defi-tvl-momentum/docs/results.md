# Results — Study 583 (DeFi-TVL-Momentum): TVL flows as a token-return signal

*Generated from [`defi_tvl_momentum/`](../defi_tvl_momentum/) on this study's **deterministic
synthetic panel** (seed 583): 60 protocols × 47 months of trailing TVL growth joined to next-month
token total returns. There is **no real tape** — a survivorship-free, point-in-time per-protocol
TVL × token-return panel is not reachable on a free, no-key stack (see
[`data.py`](../defi_tvl_momentum/data.py)), so the Signal axis is capped at **WEAK** and every
number below is the offline machinery running on a planted world. Planted-effect panel fingerprint
`0b6d95a52429`; null-world panel fingerprint `0a3c150d737e`. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The folklore: when money floods into a DeFi protocol — its **total value locked** (TVL) climbs —
the token keeps *pumping*, so a cross-sectional **TVL-momentum** sort (long the fastest-inflow
protocols, short the outflows) should print a positive forward-return spread. We cannot test this
on a real tape a retail stack can reach — the honest panel needs the tokens that later went to
**zero** (rugs, exploits, dead forks) still present and TVL measured **point-in-time**, and no
free source delivers that cleanly. So this study proves the *engine* on a deterministic synthetic
world and states the data-availability wall openly.

On the **planted-effect world** (`tvl_alpha = 0.05`) the sort works exactly as designed: the
monthly long-short spread averages **+11.8%/mo** with a one-sample *t* of **+10.9** (placebo
*p* = **0.0005**), and the pooled protocol-level slope of forward return on the TVL-growth
signal is **+0.050** (month-clustered *t* **+11.3**). On the **null world** (`tvl_alpha = 0`) the
same engine prints a flat **+0.4%/mo** (*t* **+0.33**, placebo *p* **0.75**, slope-*t* **+0.03**) —
no false signal. So the detector is faithful. But because **no real tape backs it**, the signal
cannot be certified `REAL` (that needs a robust *t* ≥ 2 on a real tape); it is `WEAK`, and the
trade is a `MIRAGE` once you charge the crypto-scale costs and the impossible short borrow on the
collapsing-TVL leg.

## Data stamp

- **Synthetic panel (planted)**: 60 protocols × 47 months = 2820 rows, `tvl_alpha = 0.05`, seed
  583, fingerprint `0b6d95a52429`
- **Synthetic panel (null)**: same shape, `tvl_alpha = 0`, seed 583, fingerprint `0a3c150d737e`
- **Real tape**: *none* — `fetch_panel()` returns empty by construction (data-availability wall)

## The TVL-momentum sort (planted world) — the engine catches the flow

| Basket (tercile, ≈18 protocols) | Next-month token return |
|---|---|
| **Top TVL growth** (fastest inflows) — LONG | (drives the +11.8%/mo spread) |
| **Bottom TVL growth** (outflows) — SHORT | |
| **Long-short spread** | **+11.8%/mo** (one-sample *t* **+10.9**, *n* = 47) |

The label-shuffle placebo *p* = **0.0005** confirms the spread sits far in the tail — the planted
flow signal is real *in this synthetic world*, which is the point of a positive control.

## The protocol-level relation (planted world)

| | value |
|---|---|
| Pooled slope (fwd_ret on monthly-z TVL growth) | **+0.050** per z-unit |
| Slope *t* (month-clustered) | **+11.3** |
| *n* (protocol-months) | 2820 |

## Robustness — stable across cuts (planted world)

| Basket fraction | Entry lag | Long-short mean | *t* |
|---|---|---|---|
| 0.2 | 0 | +13.3% | **+9.23** |
| 0.3 | 0 (headline) | +11.8% | **+10.88** |
| 0.4 | 0 | +9.3% | **+9.42** |
| 0.2 | 1 | +5.3% | +4.24 |
| 0.3 | 1 | +4.4% | +3.90 |
| 0.4 | 1 | +4.2% | +4.23 |

The signal is strongest at the same-close fill and **decays by roughly half at a one-month entry
lag** — a realistic feature of a fast on-chain flow signal (the edge is largely gone a month after
the flow is public). It survives every basket cut in the planted world.

## Costs — the trade dies on frictions and the short borrow

| | value |
|---|---|
| Gross long-short (monthly) | **+11.8%/mo** (≈ +141.7%/yr, planted) |
| Cost per rebalance | 30 bps/leg one-way × 2 legs, monthly |
| Short borrow | 800 bps/yr on the collapsing-TVL short leg |
| Net long-short (monthly) | **+10.5%/mo** (≈ +126.5%/yr) |

Even in the *planted* world the frictions are heavy (monthly rebalance × two legs × crypto-scale
spreads, plus a punitive borrow). On a real tape — where the outflow leg you'd short is precisely
the rug/exploit tail that is **impossible to borrow** and gaps to zero — the net is worse than any
gross, and the capacity is a rounding error. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `tvl_alpha` | Mean long-short *t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **+0.30** | flat — no false signal |
| 0.02 | +4.66 | effect emerging |
| 0.05 (headline) | **+11.19** | clears the bar |
| 0.10 | +22.06 | strong |

At the null the long-short *t* is ≈ 0; planting a genuine TVL→return effect drives it positive and
past +2 as it grows. The detector works — so the honest conclusion is about **data availability**,
not a broken engine.

## Why the signal cannot certify `REAL` here

1. **No survivorship-free real tape.** The real effect (if any) is dominated by the protocols that
   *died* — TVL collapses, token → zero. A panel scraped from a live free source has already
   dropped those, biasing any measured TVL-momentum spread **upward** and making it un-tradable in
   the direction the folklore claims.
2. **TVL is revised, not point-in-time.** Free per-protocol TVL history is re-based and
   re-categorised after the fact; using today's series to rank the past is look-ahead.
3. **The short leg is uninvestable.** The outflow leg is the rug/exploit tail — no borrow, no
   liquidity, gap risk to zero.

## The honest takeaway

TVL-momentum is a plausible on-chain flow story, and the engine here would catch it if it were
there: on a planted world the long-short spread is a clean *t* +10.9 and the null stays flat. But
without a survivorship-free, point-in-time real panel — which no free, no-key stack provides — the
claim cannot be certified. `WEAK` × `MIRAGE`: real machinery, no real tape, and a short leg you
could never actually borrow.
