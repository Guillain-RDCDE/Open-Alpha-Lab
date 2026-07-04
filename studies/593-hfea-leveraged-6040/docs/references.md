# References & literature map — Study 593 (HFEA — UPRO/TMF 55/45)

## The claim under test

- **The source.** "Hedgefundie" (pseudonymous Bogleheads user), *HEDGEFUNDIE's excellent
  adventure* (Bogleheads forum, Feb 2019) and *Part 2* (Aug 2019) —
  <https://www.bogleheads.org/forum/viewtopic.php?t=272007> and
  <https://www.bogleheads.org/forum/viewtopic.php?t=288192>. The recipe: hold **55% UPRO
  (3x S&P 500) / 45% TMF (3x 20+yr Treasuries)**, rebalance **quarterly**, hold for decades.
  The pitch: the two legs' *negative correlation* lets a retail investor run 3x aggregate
  leverage without a single-3x-fund's ruin profile — "leveraged diversification" compounds
  faster than the index. The thread became one of the most-read strategy threads in retail
  investing; the original poster reallocated from 40/60 to 55/45 in Aug 2019.
- **The intellectual lineage.** Lifecycle-leverage and risk-parity arguments: Ian Ayres &
  Barry Nalebuff, *Lifecycle Investing* (2010) — young investors are under-levered;
  Asness, Frazzini & Pedersen, *Leverage Aversion and Risk Parity* (FAJ 2012) — levering a
  diversified stock/bond mix historically beat concentrated equity at the same risk.

## Why the claim could be true (and when it isn't)

- **Daily-leverage mechanics.** Cheng & Madhavan (2009, *The Dynamics of Leveraged and
  Inverse Exchange-Traded Funds*, J. of Investment Management) and Avellaneda & Zhang (2010,
  *Path-Dependence of Leveraged ETF Returns*, SIAM J. Fin. Math.): a daily-reset k-x fund
  compounds `k·r` daily minus financing and a variance-drag term `k(k−1)/2·σ²` — decay is a
  *race* between drag and trend, not a law (the desk's
  [study 100 — melting-ice](../../100-melting-ice) and
  [study 61 — slow-burn](../../61-slow-burn) measure that single-fund race). **This study is
  the distinct, next-level claim:** not one fund's decay mechanics, but whether the
  *portfolio* — two anti-correlated 3x funds plus quarterly rebalancing — beats the drag.
- **The stock-bond correlation regime.** The insurance leg works only while stock-bond
  correlation is negative — a **regime, not a constant**. It was persistently negative
  roughly 2000–2021 and flipped positive in the 2022 inflation shock: see
  Campbell, Pflueger & Viceira (2020, *Macroeconomic Drivers of Bond and Equity Risks*, JPE)
  and Ilmanen (2003, *Stock-Bond Correlations*, J. Fixed Income); the desk's
  [study 579 — equity-bond-corr-flip](../../579-equity-bond-corr-flip) tests the flip itself.
  2022 was the recipe's designed-for hedge failing exactly when leverage made it lethal:
  UPRO −57%, TMF −73% in the same calendar year.
- **Borrowing/financing drag.** The funds pay roughly (k−1)×T-bill + a swap spread + expense
  ratio; in the 2010s ZIRP era that drag was historically cheap, which flattered every levered
  backtest. Frazzini & Pedersen (2014, *Betting Against Beta*, JFE) for why levered demand
  concentrates exactly where financing looks cheap.

## What we measure, and the honesty choices

- **Real funds first, synthesis second.** UPRO/TMF real total-return tapes from mid-2009; the
  2002–2009 extension uses the daily identity `3·r_index − 2·r_bill − fee/252` with the all-in
  fee **calibrated on the real-fund overlap** (terminal-NAV match) and **validated**
  (daily-return corr 0.998 / 0.997 — the same bar study 100 set). Window starts at TLT's
  2002-07 inception; ^IRX is the bill leg (FRED is unreachable from this desk).
- **A geometric claim races in logs.** "Compounds faster" = mean monthly **log**-return gap
  (HFEA − benchmark) with a Newey-West (HAC) *t* (Newey & West 1987); Sharpe races are
  excess-vs-excess (monthly, minus the compounded bill). Andrews (1991) for the lag rule.
- **The regime split is ex-ante, not snooped.** The 2022-01 split point is the claim's *own
  mechanism* (the corr flip), and the sub-period contrast carries a Welch *t* of the
  **difference** (METHODOLOGY → the inference bar: no conditional claim without uncertainty).
- **One lag, real costs.** No forecast signal exists (the rebalance calendar is known); the
  reset trades at the quarter-end close and earns from the next session. Costs one-way × NAV
  traded per reset; the funds' internal ER + financing are already inside their NAVs.

## Method lineage (the desk's shared engine)

- **Per-leg synthesis + calibration.** [`data.synth_letf` / `data.calibrate_fee`](../hfea_leveraged_6040/data.py)
  — the study-100 identity with an overlap-calibrated all-in fee.
- **Quarterly-rebalanced pair.** [`strategy.rebalanced`](../hfea_leveraged_6040/strategy.py)
  — drifting weights, quarter-end resets, one-way costs × NAV traded.
- **HAC race + regime split.** [`strategy.race` / `strategy.hac_mean_t`](../hfea_leveraged_6040/strategy.py).
- **The 2022 autopsy.** [`strategy.year_return` / `strategy.stock_bond_corr` / `strategy.drawdown_state`](../hfea_leveraged_6040/strategy.py).
- **Synthetic control.** [`data.synthetic_world`](../hfea_leveraged_6040/data.py) +
  [`strategy.synthetic_check`](../hfea_leveraged_6040/strategy.py) — the diversification
  engine planted (ρ = −0.6, bond carry) or removed (ρ = +0.6, no carry), ≥ 20 seeds averaged.

## Data sources used here

- **yfinance** daily auto-adjusted (total-return) closes: SPY, TLT, UPRO, TMF, ^IRX —
  2002-01 → 2026-06-30, cached under `_cache/hfea_prices.csv`. All headline numbers are pinned
  in [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
- Fund facts: ProShares UPRO (launched 2009-06-25, ER 0.91%) —
  <https://www.proshares.com/our-etfs/leveraged-and-inverse/upro>; Direxion TMF (launched
  2009-04-16, ER ~1.04%) — <https://www.direxion.com/product/daily-20-year-treasury-bull-bear-3x-etfs>.

## Related desk studies (the dedup map)

- [61 — slow-burn](../../61-slow-burn) and [100 — melting-ice](../../100-melting-ice): the
  **single-LETF decay mechanics** (drag vs trend, one fund at a time). **This study is
  distinct** — it tests the leveraged *portfolio allocation* claim: whether pairing two 3x
  funds with negative correlation + rebalancing beats the index, which neither sibling asks.
- [579 — equity-bond-corr-flip](../../579-equity-bond-corr-flip): the correlation regime
  itself; here that regime is the load-bearing assumption whose 2022 failure is autopsied.
- [591 — vol-managed-portfolio](../../591-vol-managed-portfolio) and
  [592 — dual-momentum-gem](../../592-dual-momentum-gem): the neighbouring "packaged
  allocation rule" claims in this lot.
