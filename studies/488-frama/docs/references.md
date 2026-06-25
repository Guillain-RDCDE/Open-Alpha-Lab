# References & literature map — Study 488 (FRAMA)

## The claim under test

- **The folklore.** FRAMA (the *Fractal Adaptive Moving Average*) is a moving average whose
  smoothing constant adapts to the **fractal dimension** of recent price: in a clean trend it
  speeds up and hugs price, in chop it slows down and flattens. The pitch, repeated on every
  charting forum, is that this "fractal-adaptive" behaviour lets a `price > FRAMA` cross-up long
  **catch trends sooner and avoid whipsaws**, beating a plain fixed-length moving average.
- **The source.** **John F. Ehlers**, *"FRAMA — Fractal Adaptive Moving Average"*, **Technical
  Analysis of Stocks & Commodities** (Sept. 2005), and his books *Rocket Science for Traders*
  (2001) and *Cybernetic Analysis for Stocks and Futures* (2004). Ehlers builds the smoothing
  constant as `alpha = exp(−4.6·(D−1))`, where `D` is the fractal dimension estimated over an
  N-bar window from the price ranges of its two halves vs the whole — the same range-ratio idea
  behind **Mandelbrot's** box-counting/`R/S` dimension and **Hurst (1951)** rescaled-range
  analysis. The N-bar two-halves estimator is Ehlers' own approximation.
- **Variants.** Ehlers later proposed a "modified" FRAMA with a slower lower bound (the `FC/SC`
  fast/slow-constant clamp) and Kaufman's **KAMA** (Adaptive Moving Average, 1995) is the older
  cousin that adapts via an *efficiency ratio* rather than a fractal dimension. All are
  **affine/recursive variants of an EMA with a state-dependent alpha** and inherit the same drift
  confound tested here.

## Why this is a "theory" / mechanical-proxy study

FRAMA is fully mechanical (no discretion), so we encode it exactly as Ehlers specifies and test
the *trading claim* honestly:

- **Causal indicator.** The FRAMA recursion uses only bars up to `t`; the fractal dimension is a
  trailing rolling window — no look-ahead. The cross-up is read on the close of `t` and the trade
  entered at the close of `t+1` (one documented lag).
- **The honest baselines.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only entry
  inherits the drift. We add a **fixed-EMA comparator** of the same average speed — the direct
  test of "does the *adaptive* part add anything over a static MA?" — and a **shuffled-alpha
  placebo** that permutes the adaptive smoothing constants in time (destroying the
  fractal-dimension link) while keeping the alpha marginal — the direct test of "is the fractal
  adaptation load-bearing?".

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *excess-vs-excess* and *signal-vs-baseline*,
  never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, **Journal of Finance**) formalize testing chart/indicator rules against a properly
  matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, **JF**) and White (2000, *A Reality Check for Data Snooping*,
  **Econometrica**) show how trend-fitted rules manufacture significance unless raced against a
  fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the FRAMA-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal FRAMA + fractal dimension.** [`strategy.frama`](../frama/strategy.py),
  [`strategy.fractal_dimension`](../frama/strategy.py) — Ehlers' recursion with the rolling D.
- **Fixed-EMA comparator + cross-up entries.** [`strategy.fixed_ema`](../frama/strategy.py),
  [`strategy.frama_cross_entries`](../frama/strategy.py),
  [`strategy.ema_cross_entries`](../frama/strategy.py).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../frama/strategy.py),
  [`strategy.hac_t`](../frama/strategy.py), [`strategy.run_experiment`](../frama/strategy.py).
- **Adaptation placebo.** [`strategy.shuffled_alpha_placebo`](../frama/strategy.py) — permute the
  per-bar alpha in time, keep its marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../frama/data.py) plants a real,
  persistent trend (knob `edge`); with `edge = 0` the detector must NOT manufacture significance —
  the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the gold-standard sibling: a
  drawn channel tested with the random-entry baseline + a geometry placebo; same None × Mirage.
- [`../../432-hull-moving-average`](../../432-hull-moving-average) and the broader moving-average /
  adaptive-MA zoo (KAMA, DEMA/TEMA) — most land None × Mirage because a smoother of past price
  re-describes the trend rather than forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; FRAMA is a clean live example of beta masquerading as an
  adaptive indicator, with the fixed-EMA comparator isolating the (absent) value of adaptation.
