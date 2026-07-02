# Results — Study 582 (ETH-Gas-Fees): gas spikes as an ETH top signal

*Generated from [`eth_gas_fees/`](../eth_gas_fees/). **This is a synthetic-only study on the signal
side**: a clean daily, survivorship-free Ethereum **gas-fee** tape needs an archive node or a keyed
API (Etherscan / Dune / Owlracle), which a no-key retail stack does not reach. So there is **no
real gas→return headline** — the numbers below are the deterministic offline core. The only real
data is the **ETH-USD** daily tape (yfinance, 2018-01-01 → 2026-06-29, 3101 daily returns, daily
vol **0.0444**, fingerprint `d4b0cdbf2e5e`) used to calibrate the synthetic return distribution
and to state the data-availability limit with a real anchor. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Top signal on-chain?" `UNPROVEN`

The folk claim (crypto Twitter / on-chain analytics): when Ethereum **gas fees** spike, the chain
is congested because everyone is minting, aping and leveraging at once — euphoric activity that
marks a **top**, so *forward* ETH returns should be LOW after a gas spike (a contrarian read). The
bullish counter-read: high gas is *genuine demand* for blockspace and is neutral-to-positive.

We **cannot certify the claim on a real tape**, because the daily gas-fee series required is out of
reach for a no-key stack. What we can do honestly is build a deterministic coupled (gas, ETH price)
world where congestion follows euphoria, run the full apparatus on it, and prove the engine is a
faithful detector — flat at the null, sharply negative when a real top signal is planted.

On the **null headline world** (fingerprint `0e87032ad333`; gas spikes carry no forward
information by construction), the top-decile gas-spike days deliver a spike-minus-calm forward
return gap of **+0.07%** (two-sample *t* **+0.17**), a HAC slope-*t* of **−0.04**, and a
label-shuffle placebo *p* = **0.856** — no signal, exactly as it must read. The contrarian
short-ETH overlay *loses* even gross (**−16.2%**/yr, HAC *t* −1.00) and worse net
(**−21.2%**/yr). So `NONE` on the signal axis (synthetic-only — no REAL tape to clear a *t* ≥ 2,
and the null run is flat), `MIRAGE` on tradability (a short-ETH-on-congestion overlay bleeds even
before the punitive crypto borrow), and `UNPROVEN` on the myth itself (the mechanism is coherent
and the engine would catch it, but no real gas tape exists here to decide it).

## Data stamp

- **Real ETH-USD** (calibration/anchor only): daily close, 2018-01-01 → 2026-06-29, 3101 returns,
  daily vol 0.0444, fingerprint `d4b0cdbf2e5e`
- **Synthetic null world** (headline): 1461 days, 1432 usable rows, fingerprint `0e87032ad333`
- **Gas-fee series**: SYNTHETIC — no real daily gas tape on a no-key retail stack (named on the
  SIGNAL axis)

## The null headline — no forward information in gas spikes (as it must read)

| | value |
|---|---|
| Spike-day (top-decile gas spike) mean forward ETH return | **+0.44%** (n = 144) |
| Calm-day mean forward ETH return | **+0.37%** (n = 1288) |
| Spike − calm gap | **+0.07%** (two-sample *t* **+0.17**) |
| Forward-return slope-*t* on gas spike (HAC) | **−0.04** |
| Label-shuffle placebo *p* | **0.856** |

A contrarian top signal would be a *negative* gap and a *negative* slope-*t*. At the null there is
neither — the placebo *p* near 0.86 confirms the tiny gap is pure noise. This is the honest "we
see nothing" reading the engine produces when there is no real signal to find.

## The contrarian overlay — loses even gross

| | value |
|---|---|
| Short-ETH-on-euphoric-congestion overlay, GROSS | **−16.2%**/yr (Sharpe −0.52, HAC *t* −1.00) |
| NET (10 bps one-way + 1000 bps/yr crypto borrow on short days) | **−21.2%**/yr (Sharpe −0.69, HAC *t* −1.30) |

Shorting ETH on gas spikes fights the coin's drift and pays a heavy crypto borrow for the
privilege. There is nothing to harvest in the null world, and no real tape to suggest otherwise.

## Robustness — the sign never lights up across horizons (null world)

| Forward horizon (days) | Spike − calm gap | Gap *t* | Slope-*t* (HAC) |
|---|---|---|---|
| 1  | +0.001 | +0.17 | −0.04 |
| 3  | +0.003 | +0.34 | +0.31 |
| 7  | +0.009 | +0.82 | +1.05 |
| 14 | +0.046 | +2.70 | +1.56 |
| 30 | +0.042 | +1.57 | +1.03 |

No horizon produces a *negative* (top-signal) slope-*t* past −2; the one gap-*t* that pokes above 2
(14-day) is the *wrong sign* for the contrarian claim and does not survive as a HAC slope. A signal
that never appears in the null — and would be a fluke if it did — is not a signal.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `top_signal_beta` | Mean forward slope-*t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.29** | flat — no false signal |
| −0.001 | −1.47 | signal emerging |
| −0.002 | **−2.65** | clears the −2 bar |
| −0.004 | −5.02 | signal clear |
| −0.006 | −7.40 | signal loud |
| −0.010 | −12.16 | signal unmistakable |

At the null the slope-*t* averages ≈ 0; planting a genuine contrarian top signal
(`top_signal_beta < 0`) drives the forward slope-*t* negative and past −2 as it grows. The detector
works — so if a real gas tape existed and carried a top signal, this engine would find it.
(Control only; never cited for a real-tape stamp — there is no real gas tape here.)

## Why this can't be certified here

1. **No real gas tape.** The signal side is synthetic. A daily, survivorship-free Ethereum
   gas-fee series back through the 2021 mania and the 2022 bear needs an archive node or a keyed
   API. A no-key retail stack reaches ETH-USD prices but not gas — so a REAL *t* ≥ 2 is impossible
   here, which caps the SIGNAL axis at `NONE` (as with the desk's lego-returns / whisky-cask /
   sneaker-resale synthetic-only studies).
2. **EIP-1559 & L2 migration change the meaning of "gas."** Post-1559 (Aug 2021) a base fee is
   burned and fees adjust algorithmically; the 2023-25 rollup migration moved most activity to L2s,
   so mainnet gas today measures a *different* thing than in 2021. Any real study must handle these
   regime breaks — a caveat that only sharpens the "no clean real tape" problem.
3. **The mechanism is contemporaneous.** Even in the coupling we model, gas and price surge
   *together* (congestion follows euphoria). Whether that congestion *leads* forward returns is
   exactly the empirical question — and it is unanswerable without the real gas tape.

## The honest takeaway

Gas-fee spikes as an ETH top signal is a **coherent, testable story with no reachable real tape**
on a retail stack. The engine is proven faithful (flat at the null, past −2 when a top signal is
planted, seed-robust over 25 seeds), and the null-world contrarian overlay loses even gross. So
`NONE` × `MIRAGE`, with the myth itself `UNPROVEN` — not busted, not confirmed, just **untested on
real data** until someone brings an archive-node gas series.
