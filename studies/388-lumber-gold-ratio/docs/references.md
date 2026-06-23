# References & literature map — Study 388 (Lumber-Gold-Ratio)

## The claim under test

- **The rule (Gayed & Bilello).** Michael A. Gayed & Charlie Bilello, *An Intermarket Approach
  to Beta Rotation* (2015; Dow Award-winning paper, SSRN). They argue the **lumber/gold ratio**
  is a clean, leading **risk-on/risk-off** gauge: lumber is cyclical (housing, construction,
  growth) while gold is defensive (fear, store of value). When **lumber outperforms gold** over a
  trailing window, tilt toward higher-beta equities; when **gold outperforms lumber**, rotate to
  defensives / long bonds. Popularised thereafter across financial media as a "smart-money"
  intermarket switch.
- **The folklore.** Repeated in market-technician writing and on finance Twitter/X as a near-magic
  regime signal — "lumber knows before stocks do." The testable, tradable distillation is a binary
  **stock/bond rotation switch** keyed off the standardised ratio level, which is exactly what we
  build and race against passive alternatives.

## Intermarket analysis — the broader tradition

- **John J. Murphy, *Intermarket Analysis* (2004)** and *Intermarket Technical Analysis* (1991) —
  the canonical text on cross-asset linkages (bonds↔stocks↔commodities↔currencies) that the
  lumber/gold idea descends from. The hope is that one market's move *leads* another's; the
  recurring empirical finding is that such leads are unstable and largely arbitraged.
- **Commodity ratios as macro gauges.** The copper/gold ratio (Dr Copper vs the fear metal) and
  the gold/oil ratio are the same genre — a cyclical commodity over a defensive one, read as a
  growth/inflation thermometer. See the related desk studies below.

## Why lumber futures are not usable to 2026 — and what we do instead

- **`LBS=F` discontinued (May 2023).** The CME random-length lumber future (Yahoo `LBS=F`), the
  series the *original* lumber/gold ratio is built from, was **discontinued in May 2023** and
  replaced by a smaller **`LBR`** contract with little free history. A current-window study to 2026
  therefore cannot use cash-lumber futures end-to-end.
- **WOOD as the lumber proxy.** We use the **WOOD** (iShares Global Timber & Forestry) ETF — a
  continuous, transparent, cache-able series — as a **lumber proxy**, named on the Signal axis. It
  is a basket of forestry/timber *equities*, not the cash lumber price, so it co-moves with stocks
  (if anything *flattering* a risk-on reading). The verdict does not hinge on this choice: the
  switch is beaten by a passive 60/40 and ties a same-exposure random control, neither of which the
  cash-lumber series would reverse.

## Why "beats 60/40" is the right bar — and the statistics

- **Excess-of-cash, not raw return.** Both sleeves (SPY, TLT) are risky, so every book is scored
  on an **excess-of-cash** daily stream (cash = ^IRX 13-week T-bill). A bond-heavy switch can post
  a flattering Sharpe in a falling-rate regime; the honest test is whether the *timing* beats the
  obvious passive mix, not whether bonds were a good place to hide.
- **HAC / Newey-West *t*.** Whitney K. Newey & Kenneth D. West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (*Econometrica*). Daily return differences are autocorrelated; the HAC *t*-stat
  ([`strategy._hac_t_mean`](../lumber_gold_ratio/strategy.py)) corrects the standard error so the
  significance of *switch-minus-benchmark* is not overstated.
- **Block bootstrap.** Hans R. Künsch (1989), *The Jackknife and the Bootstrap for General
  Stationary Observations* (*Annals of Statistics*); Politis & Romano (1992) on circular blocks.
  Resampling daily differences in **circular blocks** ([`strategy.block_bootstrap_ci`]
  (../lumber_gold_ratio/strategy.py)) preserves the autocorrelation i.i.d. resampling would
  destroy, giving an honest CI on the mean daily edge.
- **Same-exposure random control.** The cleanest folklore test: a switch that holds bonds on the
  *same number* of **random** days ([`strategy.random_positions`](../lumber_gold_ratio/strategy.py))
  has the same average beta but zero timing information. If the ratio carried regime signal, the
  real switch would beat it. It does not.

## Method lineage (the desk's shared engine)

- **The rotation switch + race.** [`strategy.switch_positions`]
  (../lumber_gold_ratio/strategy.py) (one execution lag, neutral 50/50 warm-up) and
  [`strategy.race`](../lumber_gold_ratio/strategy.py) — the head-to-head vs buy-and-hold, static
  60/40, and random rotation, all excess-of-cash and net of one-way costs.
- **Deterministic synthetic control.** [`data.synthetic_daily`]
  (../lumber_gold_ratio/data.py) plants a single forward link `pred_r` between the lagged ratio
  z-score and the next day's equity-minus-bond spread: `pred_r=0` is the null (no edge to find),
  `pred_r>0` the folklore-consistent positive control. It proves the engine has power *and* does
  not manufacture significance from noise.

## Data sources used here

- **yfinance** daily adjusted closes for **WOOD** (lumber proxy), **GLD**, **SPY**, **TLT**, and
  **^IRX** (13-week T-bill yield, the cash leg), 2008-06-25 → 2026-06-22, cached under `_cache/`
  as parquet. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 85 — Dr Copper](../../85-dr-copper/)**: copper as a growth thermometer — the same
  "cyclical commodity leads the economy" hope, tested.
- **[Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/)**: a within-metals ratio read as
  a risk gauge — sibling of the lumber/gold construct.
- **[Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/)**: defensive-over-cyclical commodity
  ratio as a macro signal — the closest cousin to this study.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: the passive static-mix
  benchmark family the lumber/gold *timing* must beat to justify itself — and (here) does not.
