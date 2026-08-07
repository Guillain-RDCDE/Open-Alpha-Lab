# Results — Study 831 (Gold Real-Yield Timing): does the real-yield *trend* time gold?

*Generated from [`gold_real_yield/`](../gold_real_yield/) over this study's cached yfinance tape: daily
adjusted close for **GLD** (SPDR Gold), **TIP** (iShares TIPS), **IEF** (iShares 7-10yr Treasury) and
the **10-year yield** (`^TNX`), **2004-11-19 → 2026-06-29**, n = **5,434** trading days. Tape
Fingerprint `4f27dc5f4b4f` (real-yield-proxy series fingerprint `ef386ee60d51`). The real-yield
signal is the TIP total-return gauge — ``ryfall = log(TIP_t) − log(TIP_{t−63})`` (>0 ⇔ real yields
fell) — ranked out-of-sample over a trailing 252-day window; the signal at close *t* trades at close
*t+1* (one-bar lag). Stamped as-of 2026-06-30 (the tape ends 2026-06-29; the partial July week is out of
scope).*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Gold ↔ real-yield inverse link" `CONFIRMED (untradable)`

Everyone "knows" gold tracks **real yields inversely**: with no coupon, gold's appeal rises as the
real return on safe assets falls, and vice versa (Erb & Harvey 2013; Baur & McDermott 2010). Two very
different claims hide inside that sentence, and this study separates them.

**The inverse *link* is real — and it is same-day.** Regressing the daily gold return on the same-day
change in the (TIP-proxied) real yield gives a correlation of **−0.26** and a HAC *t* of **−9.67** over
5,433 days: gold and real yields genuinely move in opposite directions, strongly and significantly.
That is the fact the headlines quote. But it is a **contemporaneous** relationship — to exploit it you
would need to know today's yield move before it happens.

**The tradeable *timing* version fails.** Turn the fact into a rule — "real yields have been falling →
own gold" — and the edge evaporates. Sorting forward 21-day gold returns on the (lagged) real-yield-
fall rank, the fastest-falling-yield quintile (Q5) earned **+0.93%** vs the fastest-*rising* quintile
(Q1) **+1.16%** — a Q5−Q1 spread of **−0.23%**, i.e. faintly the **wrong** direction, at HAC *t* =
**−0.36**, with a block-shuffle placebo *p* = **0.734**. The spread is wrong-signed or insignificant at
every short horizon, turns weakly *right*-signed only at 63-126 days (still sub-2: max HAC *t* +1.29 at
126d), and is tiny and unstable across sub-periods. So `NONE` on the signal axis: the *trend* of real
yields carries no forward information about gold that clears the bar. And `MIRAGE` on tradability — a
timer that owns GLD when real yields are falling ties buy-and-hold on Sharpe (**0.565 vs 0.564**) only
by sitting in cash 52% of the time; on the honest metric its mean-return spread is **−1.40 bps/day**
(HAC *t* −1.24) at **16.7** switches/year, and at 5 bps/switch its Sharpe (**0.523**) drops *below*
buy-and-hold.

The lesson is the desk's recurring one: a strong **contemporaneous** correlation is not a **predictive**
edge. Gold *co-moves* with real yields; it is not *timed* by their trend.

## Data stamp

- **Tape**: GLD + TIP + IEF + `^TNX`, daily adjusted close, 2004-11-19 → 2026-06-29, n = 5,434,
  Fingerprint `4f27dc5f4b4f`
- **Real-yield proxy** (TIP gauge): ``ryfall = log(TIP_t) − log(TIP_{t−63})`` — the trailing 63-day
  TIP total return, sign-flipped stand-in for the real-yield *change* (series fingerprint
  `ef386ee60d51` on the secondary `TNX − breakeven` level proxy, range −11.14 → +15.22, mean −2.51 —
  a cosmetic level with an arbitrary offset; only its changes carry meaning, and the headline ranks the
  TIP gauge)
- **Signal**: out-of-sample 252-day rolling percentile rank of `ryfall`, lagged one day (trades at *t+1*)
- **Survivorship**: none — single fixed ETF/yield tapes, no cross-sectional membership. Named on the
  Signal axis for completeness.
- **Window**: as-of 2026-06-30, no partial months (the tape's last bar is 2026-06-29).

## The inverse link — real, strong, and same-day (descriptive)

| Contemporaneous: gold return vs same-day real-yield change | value |
|---|---|
| Correlation | **−0.26** |
| OLS beta (gold on Δ real yield) | **−0.77** |
| Newey-West *t* on the beta | **−9.67** |
| n (days) | 5,433 |

The inverse relationship the whole story rests on **is there** — highly significant. It is also
**untradable as stated**: it is a same-day co-movement, not a lead. Everything below asks the only
question that could pay — does the real-yield *trend* predict *forward* gold — and the answer is no.

## The headline sort — the timing edge is absent (and faintly wrong-signed), 21-day horizon

| Real-yield-fall quintile (n ≈ 1,033 each) | Forward 21-day GLD return |
|---|---|
| **Q1** (real yields rising fastest) | **+1.16%** |
| **Q5** (real yields falling fastest) | **+0.93%** |
| **Q5 − Q1 spread** | **−0.23%** (HAC *t* **−0.36**, placebo *p* 0.734) |

The claim predicts Q5 > Q1 (falling yields → higher forward gold). The data give the **opposite** point
estimate, and it is statistically nothing: the block-shuffle placebo says a |spread| this size shows up
**73%** of the time by chance. No edge.

## Horizon sweep — right-signed only far out, never significant

| Forward horizon | Q5 − Q1 spread | HAC *t* |
|---|---|---|
| 5 days | −0.10% | −0.44 |
| **21 days (headline)** | **−0.23%** | **−0.36** |
| 63 days | +1.24% | +0.67 |
| 126 days | +3.32% | +1.29 |

The spread flips to the *claimed* sign only at 63-126 days and grows with horizon — but the HAC
correction (overlapping returns) keeps the *t* well under 2 (peak +1.29). Directionally the effect is
absent short-term and, at best, a faint slow drift long-term that cannot be certified.

## Lookback sweep — no lookback rescues it

| Real-yield-fall lookback | Q5 − Q1 spread (21d) | HAC *t* |
|---|---|---|
| 21 days | +0.23% | +0.31 |
| **63 days (headline)** | **−0.23%** | **−0.36** |
| 126 days | +0.44% | +0.52 |
| 252 days | +0.44% | +0.55 |

Whether the real-yield trend is measured over a month or a year, the forward-gold spread hovers around
zero (|HAC *t*| ≤ 0.55). The signal is not a matter of tuning the window.

## Sub-period sweep — tiny and unstable

| Period | Q5 − Q1 spread (21d) | HAC *t* | Reads as |
|---|---|---|---|
| 2004-2009 | −0.93% | −0.41 | wrong sign |
| 2010-2015 | +0.11% | +0.09 | nothing |
| 2016-2020 | +0.23% | +0.17 | nothing |
| 2021-2026 | +0.32% | +0.16 | faint/right, insignificant |

No window clears *t* = 2; the sign wanders and the magnitudes are a fraction of a percent. Even the
inflation-scare 2021-2026 window — where a real-yield timing rule should shine — musters only +0.32%
at *t* +0.16.

## Timing overlay — a Sharpe tie by cash-drag, a loss on mean return

| | value |
|---|---|
| Timer net Sharpe (own GLD when real yields falling, else cash; 2 bps/switch) | **0.565** |
| Buy-and-hold GLD Sharpe | **0.564** |
| Timer net Sharpe @ 5 bps/switch | **0.523** |
| Switches / year | **16.7** |
| Days invested (fraction) | **0.479** |
| Timer − buy-and-hold mean-return spread | **−1.40 bps/day** (HAC *t* −1.24) |

The timer's Sharpe *ties* buy-and-hold at 2 bps only because it is in cash 52% of the time (a
volatility-reduction artefact, not alpha); on the honest metric — the *mean-return* spread — it
**loses** 1.40 bps/day, and it churns ~17 round-trips a year, so at a realistic 5 bps/switch its Sharpe
falls *below* buy-and-hold (0.523 < 0.564). `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `edge` | Mean Q5−Q1 spread (21d) | Mean HAC *t* (25 seeds) | |
|---|---|---|---|
| 0.000 (null) | +0.03% | **+0.00** | flat — no false signal |
| 0.005 | +6.80% | +6.56 | edge emerging |
| 0.010 | +13.59% | +11.36 | clears the bar |
| 0.020 | +27.27% | +16.36 | strong |
| 0.040 | +55.43% | +18.00 | very strong |

The synthetic world carries the *same* contemporaneous inverse link (via `link_beta`) as the real tape,
so the null (`edge = 0`) is exactly the real-world case: an inverse co-movement present, a *timing* edge
absent. At the null the mean HAC *t* is ≈ 0 (no false positive); planting a genuine forward timing edge
drives the Q5−Q1 spread positive and the *t* far past 2. The detector works — so the real-tape null is a
statement about **this tape**, not a broken engine. (Control only; never cited for the real-tape stamp.)

## Why it doesn't certify here

1. **Contemporaneous ≠ predictive.** The −0.26 same-day correlation is genuine and strong, but it is a
   *co-movement*: gold and real yields react to the same shocks at the same time. Lagging the real-yield
   trend by a day to make it *tradeable* removes exactly the information that produced the correlation.
2. **Overlapping-return autocorrelation.** The only right-signed spreads appear at 63-126 days, where
   overlapping forward windows inflate the raw magnitude; the Newey-West correction rightly deflates the
   *t* below 2 (peak +1.29).
3. **A proxy, not the DFII10 tape.** The official 10-year TIPS real yield is a FRED series, not a Yahoo
   ticker; the TIP total-return gauge is the cheapest honest stand-in. But the failure here is not a
   proxy artefact — the proxy *does* deliver the strong contemporaneous link; it is the *forward* edge
   that is absent.

## The honest takeaway

Gold really does move inversely with real yields — the contemporaneous correlation (−0.26, HAC *t* −9.67)
is one of the more reliable macro co-movements on the tape. But that is a **same-day** fact, not a
crystal ball: the real-yield *trend* does not predict *forward* gold (Q5−Q1 spread −0.23%, HAC *t* −0.36,
placebo *p* 0.73, wrong-signed at the headline horizon), and a costed timer built on it ties buy-and-hold
by cash-drag and loses on mean return after ~17 switches a year. `NONE` × `MIRAGE`, with the inverse link
`CONFIRMED but untradable`. The synthetic control confirms the engine would bank a real timing edge — so
this is the tape talking, not the code.
