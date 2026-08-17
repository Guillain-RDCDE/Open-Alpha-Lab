# Results — Study 934 (Lump Sum vs DCA) on the real daily tape

*Every number below is transcribed from one run of [`examples/verify.py`](../examples/verify.py)
(cache-only, offline, exit 0) — the script prints, this file records. The experiment: $1 of new
money, twelve months, two ways in. **Lump sum** buys the whole dollar at the first
execution date. **DCA** buys 1/12 at each of twelve consecutive month-ends; the balance
still waiting sits in **BIL** and earns its actual total return. Both arms are valued on
the **same** terminal date. Every purchase — the single lump-sum buy and each of the
twelve tranches — is decided at a month-end close and executed at the **next trading
day's close** (one execution lag, no look-ahead). Costs are one-way × NAV, 1 bp headline;
neither arm ever shorts, so there is no borrow leg. Daily **total-return** closes
(`yfinance`, `auto_adjust=True`). The race is rolled over **every start month** of the
SPY∩BIL sample. As-of **2026-06-30** (the partial current month is dropped; a window is
kept only if its twelfth month-end lands on or before the as-of).*

## Data stamp

| Legs | Window | n days | n windows | Fingerprint |
|---|---|---|--:|---|
| SPY (equity), BIL (cash) | 2007-05-30 → 2026-06-30 | 4,802 | 217 | `edef65f148a6` |
| IEF (bonds), BIL (cash) | 2007-05-30 → 2026-06-30 | 4,802 | 217 | `9803a2a6157d` |

Start months run 2007-06-01 → 2025-06 (last window ends 2026-06-01). BIL's 2007 inception
gates the honest window: before it there is no live T-bill ETF to credit the waiting cash
with.

**A caveat on the fingerprints.** `studies/_cache` is shared with every other study on the
desk, and other studies re-pull the same tickers on their own schedules and with their own
start dates — the SPY and BIL parquets were rewritten several times during this build day
(SPY/BIL fingerprints `d92041f54285` → `0934e3a746e3` → `9cce1b76d021` → `edef65f148a6`),
and the cache's SPY history now begins **2000-01-03**, not at SPY's 1993 inception. Two
consequences, both handled rather than hidden:

1. The fingerprints above identify the pull that produced this file and **will move** under
   the next refresh. What did *not* move across four refreshes: every number printed below
   — 76.0%, +5.05c, HAC *t* +3.19, the exposure-matched −0.04c — to the last decimal. A
   fingerprint change with unchanged reads is the cache being re-fetched, not the study
   being re-tuned.
2. The long-history extension is floored at a **pinned** `LONG_START = 2000-01-03`, so it
   reproduces whether the cached SPY tape starts in 1993 or in 2000. It is not this study's
   place to assume how deep somebody else's fetch went.

## The headline — SPY, 12-month horizon, 12 monthly tranches

| Read | Value |
|---|--:|
| Lump-sum win rate | **76.0%** (Wilson 95% CI [69.9%, 81.2%]) |
| Mean gap, lump − DCA | **+5.05 cents per $1** |
| Median gap | **+6.09 cents** |
| HAC *t* on the mean gap (12 lags) | **+3.19** |
| Non-overlapping *t* (every 12th start, all 12 phases) | **+2.18** (phase range +1.26 … +2.91) |
| Block-bootstrap 95% CI on the mean gap | **[+1.57, +7.90]**, share < 0 = 0.2% |

The windows overlap by construction (monthly starts, twelve-month horizons), so the naive
*t* would be badly overstated; both the HAC correction and the fully non-overlapping check
clear |*t*| = 2, and the bootstrap interval excludes zero. **The terminal-wealth gap is
real. The next section is about what it is made of, and the answer is not timing.**

## The gap is EXPOSURE, not timing — the control that decides the Tradability stamp

A DCA schedule is not merely *later* into the market; it is *less* in it. Tranche *j* is
invested for only (12 − *j*)/12 of the window, so DCA's time-weighted equity exposure is
**(n+1)/2n = 13/24 = 54.2%** — an analytic number that falls out of the schedule, with no
hindsight and nothing fitted. Racing it against a **full-beta** lump sum and calling the
difference an advantage is a beta race, not a timing test.

The honest control: race DCA against a **static** portfolio holding that same 54.2% in SPY
and 45.8% in the same BIL leg for the whole twelve months — same average exposure, same
terminal date, no schedule at all.

| Comparison (SPY) | Mean gap | Win rate | HAC *t* | Non-overlap *t* | Bootstrap 95% CI |
|---|--:|--:|--:|--:|---|
| Lump sum vs DCA (headline) | **+5.05c** | 76.0% | **+3.19** | +2.18 | [+1.57, +7.90] |
| **54.2% static vs DCA** (analytic weight, no hindsight) | **−0.04c** | 53.5% | **−0.08** | −0.06 | **[−1.18, +0.98]** |
| 58.4% static vs DCA (dispersion-matched weight, **in-sample fit**) | +0.43c | 58.1% | +0.78 | +0.32 | [−0.80, +1.48] |
| Same control on the 2000-2026 long tape | +0.03c | — | +0.06 | — | [−0.86, +0.82] |

**Every cent of the headline is the extra beta.** Hold DCA's own average exposure for
twelve months and you land exactly where DCA lands (−0.04 cents, *t* = −0.08, an interval
tight around zero on a sample where the raw gap was +5.05c). The same read on risk-adjusted
terms, with **both** legs excess of the **same** cash leg: the lump sum earns +11.11% per
window at 0.651 per unit of dispersion, DCA earns +6.06% at **0.608** — the two arms are
paid at very nearly the same rate for the risk they take, and the lump sum's extra money is
the extra risk.

## The full dispersion — where each arm actually lands

| | Lump sum | DCA |
|---|--:|--:|
| Mean 12-month outcome | **+12.35%** | +7.29% |
| Worst window | **−45.85%** | **−36.13%** |
| SD of terminal wealth | 0.1701 | **0.0999** |

- **Dispersion ratio (DCA ÷ lump) = 0.587.** The risk half of the folklore is *true*: DCA
  lands in a band a little over half as wide, and its worst twelve months are ~10 pp
  shallower. It is the *return* half that fails — and the reason both halves happen at
  once is a single fact: DCA owns about half as much stock.
- The gap itself is wide-tailed: p5 **−10.5c**, p95 **+18.7c**, worst **−30.7c** (a start
  month just before 2008), best **+43.9c**. Going all in is right three times in four —
  and when it is wrong it is expensively wrong.

## Is there a state where the advice comes good?

Terciles are cut on the in-sample distribution of starts, so this is a *hindsight* cut —
it asks whether such a state existed, not whether you could have traded it live. "Stretch"
is a **price-based PROXY** for expensive (level vs trailing three-year mean); no earnings
data is used anywhere in this study. The trailing mean needs a three-year runway inside the
sample, so the terciles cover **181 of the 217** starts (from 2010-06 on); the drawdown cut
uses all 217.

| Start state | n | Lump win rate | Mean gap | HAC *t* | Mean 12m SPY return |
|---|--:|--:|--:|--:|--:|
| Cheap tercile | 61 | 93.4% | +10.10c | +9.66 | +22.1% |
| Middle tercile | 60 | 80.0% | +5.64c | +5.22 | +13.8% |
| **Stretched tercile** | 60 | **73.3%** | **+3.43c** | +2.38 | +8.5% |
| **Starting ≥10% below the high** | 59 | **72.9%** | +5.26c | **+1.50** | +13.8% |
| Starting near the highs | 158 | 77.2% | +4.98c | +3.63 | +11.8% |

**There is no state in which DCA wins on average.** The advantage shrinks where the story
says it should — from +10.1c starting cheap to +3.4c starting stretched — but it never
crosses zero, and it is *not* larger inside a drawdown (the drawdown cut is the one place
the *t* falls below 2, on 59 windows). Read the last column: the gap tracks the realised
12-month SPY return almost one-for-one, which is the same message as the exposure control.
Buying the fear is not what DCA does; it just holds less equity for longer.

## Robustness

**Two eras (split by start month, 2017-01-01).**

| Era | n | Win rate | Mean gap | HAC *t* |
|---|--:|--:|--:|--:|
| 2007-06 → 2016-12 | 115 | 70.4% | +4.09c | +1.65 |
| 2017-01 → 2025-06 | 102 | 82.4% | +6.14c | +3.39 |

Same sign in both halves, but the early half's own *t* sits below 2 — and the long history
below says that is **not** only a sample-length artefact.

**Long-history check — SPY from the pinned `LONG_START = 2000-01-03` (0% cash ASSUMPTION;
BIL does not exist before 2007).** 305 windows (2000-02 → 2026-06), win rate **74.4%**,
mean gap **+4.47c**, HAC *t* **+3.25**, non-overlapping *t* **+2.36**. Decade by decade:

| Start decade | n | Win rate | Mean gap | HAC *t* | Mean 12m SPY |
|---|--:|--:|--:|--:|--:|
| **2000s** | 119 | **60.5%** | **+0.67c** | **+0.26** | **+1.6%** |
| 2010s | 120 | 85.0% | +6.10c | +6.61 | +13.5% |
| 2020s | 66 | 80.3% | +8.36c | +3.06 | +17.6% |

This is the most honest table in the study. Across the lost decade — the one ten-year
stretch in this sample where US equities paid essentially nothing — **the lump sum's
advantage is +0.67 cents with a *t* of +0.26**: gone, not merely smaller. The advantage is
the size of the risk premium that actually showed up, no more and no less, which is exactly
what the exposure control predicts (the exposure-matched gap on this long tape is +0.03c,
*t* = +0.06).

*An earlier draft of this file reported a 1993-start extension with 389 windows and an
early half of t = +2.19, and used it to argue the early softness was a sample-length
artefact. That run is not reproducible — the shared cache's SPY history now starts in 2000
— and the claim it supported was wrong. Both are corrected here.*

**Cost sweep (proportional, one-way bps of NAV).** 0 / 1 / 5 / 25 bps → mean gap +5.05 /
+5.05 / +5.05 / **+5.04c**. This is not a rounding failure: both arms put the **same $1**
to work in total, so a proportional cost is charged on the same notional in each and
cancels in the difference. The cost that does *not* cancel is the fixed ticket.

**Fixed-ticket sweep (ASSUMPTIONS: a $10,000 windfall, DCA pays 12 tickets, the lump sum
one).** $0 / $1 / $5 / $10 per trade → mean gap +5.05 / +5.16 / +5.60 / **+6.15c**. Every
dollar of commission only widens the lump sum's lead.

**Cash-leg assumption.** With idle cash earning a flat **0%** (the popular write-ups' and
Study 101's assumption) the gap is **+5.64c** (*t* = +3.57) rather than +5.05c: crediting
DCA with the real T-bill path over 2007-2026 gives back **0.6 cents** of the lump sum's
lead — real, and far too small to change the answer.

**Tranche sweep.** 3 / 6 / 12 / 24 monthly tranches → mean gap +0.81 / +2.18 / +5.05 /
**+11.61c**, win rate 65.0% / 70.9% / 76.0% / 86.8%, dispersion ratio 0.727 / 0.639 /
0.587 / 0.497. Return and dispersion move together at every point on the line, because the
only thing the tranche count changes is average exposure ((n+1)/2n: 66.7% / 58.3% / 54.2% /
52.1%).

**Selection / survivorship, named.** One tape, and the one that won: US large-cap equity,
2007-2026 (and 2000-2026 on the long check). Nothing here is a cross-sectional panel, so
there is no membership survivorship — but the *tape* is a choice, and it is the tape on
which the equity premium showed up. The 2000s row above is the in-sample version of that
warning; the IEF variant below is the cross-asset version. Neither the result nor its sign
is portable to a market whose premium fails to arrive.

## The bond-heavy variant — IEF instead of SPY

| Read | Value |
|---|--:|
| Lump-sum win rate | 59.0% (Wilson 95% CI [52.3%, 65.3%]) |
| Mean gap | **+1.02 cents per $1** |
| HAC *t* | **+1.45** (non-overlapping *t* +1.11) |
| Bootstrap 95% CI | **[−0.36, +2.29]** — includes zero |
| Dispersion ratio | 0.628 |
| Exposure-matched gap (54.2% static vs DCA) | +0.08c, *t* = +0.47, CI [−0.26, +0.42] |

On a 7-10y Treasury sleeve the lump sum's advantage is **one fifth** the size and **not
statistically distinguishable from zero**. That is the mechanism showing its face: the
prize is the risk premium of whatever you are buying, collected earlier. Thin premium,
thin prize — and matched for exposure, no prize at all.

## Synthetic control (machinery proof only — never supports the stamp)

Twelve independent 25-year paths per world. A single 16%-vol path is a very noisy estimate
of its own drift, so the control is read across seeds, not from one draw.

| Planted world | Mean gap (sd across seeds) | Mean win rate | Lump wins on |
|---|--:|--:|--:|
| Fat excess drift (`signal_strength=+1`) | **+3.74c** (1.98) | 62.7% | **12/12 seeds** |
| Null (`signal_strength=0`) | **+0.13c** (1.80) | 46.6% | 6/12 seeds |
| Falling market (`signal_strength=−1`) | **−3.15c** (1.63) | 32.0% | **0/12 seeds** |

The harness crowns the lump sum only where a premium is planted, sits on zero on the null,
and crowns **DCA** when the tape falls. It has no thumb on the scale. The exposure control
is run on all three synthetic worlds too (in the test-suite): whatever drift is planted,
matching the average exposure collapses the gap to under a cent — the invariant that says
the headline is a beta difference in every world, not only in this one.

## Verdict

- **Signal — Real.** The lump sum finishes richer in **76.0%** of the 217 start months
  (Wilson CI clear of one half) and by **+5.05 cents per invested dollar** on average, with
  an overlap-corrected HAC *t* of **+3.19**, a non-overlapping *t* of **+2.18** (every
  phase positive), and a bootstrap CI of **[+1.57, +7.90]** that excludes zero. The sign
  holds in every conditional cut, in both eras, at every cost, and on the 2000-2026 long
  history (*t* = +3.25). The direction is the opposite of the advice: averaging in costs
  money. What the stamp does *not* say is that it costs money for the reason the advice
  fights about — see the next line.
- **Tradability — Mirage.** Not because costs eat it (they cancel), but because there is
  nothing to eat: matched for exposure, the gap is **−0.04 cents** with *t* = **−0.08** and
  a bootstrap CI of **[−1.18, +0.98]**, and the two arms earn statistically the same reward
  per unit of dispersion (0.651 vs 0.608, both excess of the same cash leg). This is the
  desk's textbook Mirage — *"it's just beta you were always paid for"*. Two more nails: the
  advantage vanishes in the one decade the premium did not arrive (2000s, +0.67c, *t* =
  +0.26), and it is statistically invisible on a bond sleeve (IEF, +1.02c, CI through
  zero). Acting on it is still free and still correct **if** you have already chosen your
  equity weight — but the money it "makes" is compensation for risk you decided to take
  earlier, not an edge you can bank, scale, or repeat.
- **Does DCA lower risk? — Confirmed.** Dispersion ratio **0.587** and a worst window of
  −36.1% against the lump sum's −45.9%. Real — and bought with average exposure, not with
  better prices: the 54.2% static portfolio has the same dispersion (0.0926 vs 0.0999) and
  the same terminal wealth, without the schedule. If you want DCA's calm, own DCA's weight;
  the twelve tranches add nothing to it.
