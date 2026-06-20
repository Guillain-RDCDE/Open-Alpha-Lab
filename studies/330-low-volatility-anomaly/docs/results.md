# Results — Study 330 (Low-Volatility-Anomaly): SPLV vs SPHB, 2011–2026

*Generated from [`low_volatility_anomaly/`](../low_volatility_anomaly/) over the cached monthly
total-return tape of **SPLV** (Invesco S&P 500 Low Volatility), **SPHB** (Invesco S&P 500 High
Beta) and **SPY** (market), Yahoo. Common window **2011-06 → 2026-05**, 180 months (the partial
2026-06 bar is dropped). Inputs fingerprint `a5ce034427ab`.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE` · "Boring beats exciting, risk-adjusted?" `CONFIRMED`

The low-volatility anomaly (Baker–Bradley–Wurgler 2011; Frazzini–Pedersen 2014) says calm
stocks out-earn wild ones *per unit of risk*. Its most literal retail expression is the
**SPLV vs SPHB** ETF pair — the two opposite tails of the same S&P 500, as funds you can
actually hold. Over 2011–2026 the **ranking is confirmed**: SPLV's Sharpe (**0.85**) sits above
SPHB's (**0.66**), a +0.19 gap, on a fraction of the risk. But the **tradable long-short does
not certify**: a beta-neutral long-SPLV / short-SPHB book earns +3.5%/yr at HAC *t* = **1.43**
— short of the *t* ≥ 2 bar — and the *naive* dollar-neutral version **loses −6.4%/yr** because
the high-beta leg simply earns more beta return in an up-market. So `WEAK` on the signal axis
(the Sharpe ranking is the right way round, but this tape alone can't bank the spread),
`FRAGILE` on tradability (the salvageable piece is a long-only low-vol *defensive tilt*, not a
self-financing money machine), and `CONFIRMED` on the qualitative "boring beats exciting"
question — risk-adjusted, it does.

## Data stamp

- **Pair**: SPLV (low-vol) / SPHB (high-beta) / SPY (market), monthly total return,
  2011-06 → 2026-05, 180 months, fingerprint `a5ce034427ab`

## The legs — boring wins on Sharpe, loses on raw return

| | CAGR | Sharpe | vol | max-DD |
|---|---|---|---|---|
| SPLV (low-vol) | +9.7% | **0.85** | **11.7%** | **−21%** |
| SPHB (high-beta) | +14.2% | 0.66 | 24.8% | −37% |
| SPY (market) | +14.2% | 0.98 | 14.6% | −24% |

SPLV took **less than half** the volatility of SPHB and a much shallower drawdown, and despite
trailing on raw return it delivered a **higher Sharpe** (0.85 vs 0.66) — the anomaly's
signature. Note that SPLV did *not* beat the **market** on Sharpe (0.85 vs 0.98) in this
bull-dominated sample, the same caveat [58 Bunker](../../58-bunker/) records for USMV.

## The tradable spreads

| Book | mean (ann) | Sharpe | HAC *t* |
|---|---|---|---|
| Raw dollar-neutral (long SPLV, short SPHB) | **−6.4%/yr** | −0.31 | −1.24 |
| Beta-neutral (long SPLV, short β-matched SPHB), gross | **+3.5%/yr** | 0.34 | **1.43** |
| Beta-neutral, net (5 bps/leg + 50 bps/yr borrow) | +2.5%/yr | — | 1.02 |

- The **raw** long-low/short-high trade is negative by construction: shorting the high-beta leg
  means shorting the decade's beta winner.
- **Beta-neutralising** (hedging out the structural SPLV≈0.7 / SPHB≈1.4 gap) lifts the mean to
  +3.5%/yr, market beta ≈ 0.0 — but the HAC *t* is **1.43**, with a block-bootstrap 95% CI on
  the annual mean of **[−0.9%, +8.1%]** that straddles zero. Below the inference bar.
- Costs and a modest short borrow take the net book to +2.5%/yr (*t* 1.02): what looked like a
  thin edge is mostly gone once you pay to run it.

## Why the spread can't be certified

1. **A bull-dominated sample.** 2011–2026 was a high-beta regime; the low-vol anomaly earns its
   keep in *bear* markets and over full cycles, so the window understates it (the mirror of why
   high-beta won in [43 Free-Lunch](../../43-free-lunch/) / [53 Jackpot](../../53-jackpot/)).
2. **Short n.** SPLV/SPHB both launched 2011-05, leaving only ~15 years of monthly data — too
   few months for a thin Sharpe gap to clear *t* = 2.
3. **The edge is defensive, not self-financing.** What survives is the long-only SPLV tilt
   (lower vol, shallower drawdown, a competitive Sharpe), not the dollar- or beta-neutral
   long-short — which is what real low-vol funds quietly sell.

## The honest takeaway

Boring **does** beat exciting risk-adjusted — SPLV's Sharpe sits clearly above SPHB's (0.85 vs
0.66, `CONFIRMED`). But the *tradable* embodiment of that fact is `WEAK`/`FRAGILE`: the
beta-neutral spread that would monetise it earns only +3.5%/yr at HAC *t* 1.43 (CI straddles
zero), the naive trade loses to the high-beta leg's beta, and costs plus borrow erode what
little is left. A real defensive Sharpe tilt, not a free lunch — and not certifiable as a
long-short on this 15-year bull sample.
