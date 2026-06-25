# References & literature map — Study 490 (Arms Index / TRIN)

## The claim under test

- **The folklore.** The **Arms Index** (a.k.a. **TRIN**, the *Trading Index*) is
  `(advancing issues / declining issues) / (advancing volume / declining volume)`. A reading
  **above 1** means volume is crowding into decliners (selling pressure); a **spike to ~2+** is
  read as a panic/washout that marks a short-term **bottom** and precedes a bounce, while a
  reading **below ~0.7** marks euphoria. The contrarian rule taught on every technician site:
  *buy the panic — when TRIN spikes high, a rebound is near.*
- **The source.** **Richard W. Arms Jr.** introduced the index in *Barron's* in **1967** and
  developed it across his books *Profits in Volume* (1971) and *The Arms Index (TRIN)* (1989). It
  is one of the oldest market-internals/breadth gauges and is still published intraday by the NYSE
  and quoted on every terminal (`$TRIN`, `$TRINQ`).
- **The mechanism claimed.** TRIN normalises *breadth* (how many names move) by *volume* (how much
  conviction backs the move). A high TRIN = many decliners hogging disproportionate volume = a
  capitulation that, by contrarian logic, exhausts sellers and reverts up.

## Why this is a breadth-**proxy** study (and what that caps)

True TRIN needs *exchange-wide* advance/decline issue counts and their volumes — a feed that is
not available offline. Following the desk's design for breadth studies, we build the tightest
mechanical proxy a proponent would accept and state the irreducible limitation explicitly:

- **5-issue breadth proxy.** Each cached ETF (SPY, QQQ, IWM, DIA, GLD) is one "issue"; it advances
  if its daily return > 0; |return| stands in for that issue's volume. A small volume floor +
  Laplace count prior keep tiny-move days from blowing the ratio to infinity. The top-TRIN days
  this proxy flags are the *real* washouts (16 & 12 Mar 2020, 1 Dec 2008), so it is directionally
  sane — but a 5-issue measure is far noisier than 3000 NYSE issues, and the volume proxy is
  magnitude, not shares. This is the **cap on the test**, stated up front.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch, hold), because *any* dip-buy inherits the
  drift. We add a **shuffled-TRIN timing placebo** that permutes when the panic days fall while
  keeping the marginal — the direct test of "does the *timing* of high-TRIN days matter?"

A richer **sector-ETF basket** (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, SPY) is wired into
`data.load_basket` for anyone with a network connection; it caches on first fetch and then serves
offline. The gate/CI run on the cached 5-name basket so they never touch the network.

## Why a high one-sample *t* is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
  Here the one-sample *t*'s (20d +3.71, 60d +3.79) look strong but largely reflect the drift —
  which is why the verdict turns on the vs-random Welch test, where the edge is only +1.53.
- **Volatility-rebound confound.** "Buy after a crash day" captures a generic short-horizon
  mean-reversion of volatility (a cousin of the VRP and short-vol literature); much of the panic
  premium here is *that*, not TRIN's breadth/volume normalisation specifically.
- **Data snooping on market-internals.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalise testing chart/technical signals against a properly
  matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, *Journal of Finance*) and White (2000, *A Reality Check for
  Data Snooping*, *Econometrica*) show how fitted rules manufacture significance unless raced
  against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the panic-vs-random difference.

## Method lineage (the desk's shared engine)

- **Breadth-proxy TRIN + panic entry.** [`strategy.compute_trin`](../arms_index_trin/strategy.py),
  [`strategy.panic_entries`](../arms_index_trin/strategy.py) — the mechanical breadth gauge with
  the close-of-t / enter-next-close lag.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../arms_index_trin/strategy.py),
  [`strategy.hac_t`](../arms_index_trin/strategy.py), [`strategy.run_experiment`](../arms_index_trin/strategy.py).
- **Timing placebo.** [`strategy.shuffled_trin_placebo`](../arms_index_trin/strategy.py) —
  permute the TRIN series in time, keep its marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../arms_index_trin/data.py) plants
  a real post-panic bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for the SPY/QQQ/IWM/DIA/GLD basket,
  2005-01-04 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../375-vxx-roll-decay`](../../375-vxx-roll-decay) and the short-vol / VRP family — the
  volatility-rebound that powers much of the panic premium here.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the same "fade the extreme"
  contrarian idiom tested with the random-entry baseline.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling technical study whose
  geometry placebo is the direct analogue of this study's timing placebo; it lands a clean
  None × Mirage, whereas TRIN's placebo is significant — a useful contrast.
- The **research-method demos** (data-mining-roulette, look-ahead, multiple-testing) frame why a
  signal-vs-zero *t* is not enough, and why a positive-but-sub-2 vs-random *t* is a Weak/Fragile
  hint, not a Real edge.
