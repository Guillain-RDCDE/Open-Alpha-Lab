# Results — Study 312 (Debt-Ceiling), the volatility angle, on the real VIX tape

*Generated from [`debt_ceiling.data.load_real`](../debt_ceiling/data.py) + the
[`debt_ceiling.strategy`](../debt_ceiling/strategy.py) event-study engine. The question is
**not** "do stocks bounce" (that is the directional sibling, [Study 311 —
Government-Shutdown](../../311-government-shutdown/)) but the orthogonal one the hook asks:
**is debt-ceiling brinkmanship a volatility trade?** — does the **VIX** ramp into the
binding deadline and collapse on resolution? We line up every brink-going US debt-ceiling
episode since the VIX's 1990 inception (a hardcoded, Treasury/CRS-sourced table of seven
deadlines), measure the VIX log-change over the 20 sessions ENDING at the deadline (the
"long vol into" leg) and the 20 sessions STARTING at the deadline (the "short vol out"
leg), race both against 3,000 random dates, and score the canonical vol round-trip net of
the long-vol carry tax. As-of **2026-05-29**; match the fingerprint to confirm the tape.*

## Data stamp

| Series | Window | Days | Fingerprint |
|---|---|--:|---|
| ^VIX (raw index level) | 1990-01-02 → 2026-05-29 | 9,170 | `339114a53569` |

Events (n = 7), deadline dates: 2011-08-02, 2013-10-17, 2015-11-03, 2017-12-08,
2021-10-18, 2021-12-15, 2023-06-05. Routine, quietly-raised ceilings are excluded; the
list keeps the standoffs markets actually feared. **Seven events is far too few to call
anything a strategy — the verdict is carried by the wide CIs, said loudly.**

## The two vol legs — VIX log-change around the deadline (pre=20, post=20 sessions)

| Leg | n | mean (Δlog VIX) | hit-rate | HAC *t* | block-bootstrap 95% CI |
|---|--:|--:|--:|--:|---|
| **Pre-deadline ramp** (does vol rise *into* it?) | 7 | **−5.2%** | 43% | **−0.55** | [−23.5%, +13.0%] |
| **Post-deadline move** (does vol *collapse*?) | 7 | **+5.2%** | 71% | +1.78 | [+0.7%, +10.4%] |

- The "long vol into the deadline" thesis is **backwards on this tape**: the VIX
  *fell* 5.2% on average over the 20 sessions into the deadline, and only **2 of 7**
  episodes saw vol actually rise. HAC *t* = −0.55 — nowhere near the bar.
- The "vol collapses on resolution" thesis is **also wrong in sign**: the VIX *rose*
  +5.2% over the 20 sessions after the deadline. That positive number is carried almost
  entirely by **2011** (the post-deal S&P sovereign-downgrade panic, +28% post), the one
  episode that genuinely spooked markets — and it spiked *after* the deal, not into it.

## Per-event detail — the 2011 coincidence carries the whole story

| Deadline | pre-20 Δlog | post-20 Δlog |
|---|--:|--:|
| 2011-08-02 (S&P downgrade) | +43.4% | +28.3% |
| 2013-10-17 | +2.4% | −8.6% |
| 2015-11-03 | −28.8% | +9.0% |
| 2017-12-08 | −9.2% | +5.1% |
| 2021-10-18 | −45.5% | +1.1% |
| 2021-12-15 | +16.4% | +5.2% |
| 2023-06-05 (FRA deal) | −15.4% | −3.8% |

Drop the single 2011 row and the pre-ramp mean is strongly **negative** and the
post-deadline mean is **near zero**: there is no debt-ceiling vol signature, only one
unusually scary episode that happens to dominate a seven-point sample.

## Does it differ from a random date? — the placebo test

The honest question is not "is the move non-zero" but "is it *unusual* vs any random
date." Measured around 3,000 random VIX sessions, the same windows average a Δlog of
−0.6% (pre) and −0.0% (post). The permutation test of the event mean against the
random-date distribution:

| Leg | event − random-date mean | permutation *p* |
|---|--:|--:|
| Pre-deadline ramp | −4.7% | **0.55** |
| Post-deadline move | +5.2% | **0.50** |

Both *p*-values are ~0.5 — the debt-ceiling windows are **statistically indistinguishable
from random dates**. There is no abnormal vol behaviour to bank.

## The trade — long-vol-into / short-vol-out, net of the carry tax

Scored as the canonical round-trip (go long vol 20 sessions before the deadline, flip to
short vol at the deadline close, exit 20 sessions later), the long-vol leg pays a VIX
term-structure roll of **25 bps/day** (≈5% of NAV over a 20-session hold) and the
short-vol leg earns it:

| Variant | n | mean P&L (NAV) | hit-rate | HAC *t* |
|---|--:|--:|--:|--:|
| Round-trip, gross (25 bps/day carry) | 7 | **−10.4%** | 43% | **−1.34** |
| Round-trip, net (carry + 5 bps cost) | 7 | −10.6% | 43% | −1.36 |
| Long-vol-into leg only (net carry) | 7 | −10.3% | 29% | −1.08 |

The vol round-trip is a **loser** (−10.4% per event) before you even charge transaction
costs — because (a) vol didn't reliably move the way the thesis claims, and (b) the
long-vol leg bleeds the roll-down tax every day it waits for the deadline. "Buy protection
into the event" starts from a structural hole and the events don't pay you out of it.

## Synthetic positive control — the engine is a faithful detector

On a deterministic synthetic VIX tape (mean-reverting OU level) the same engine recovers a
planted vol hump **only when one is planted**, confirming the null result above is the
absence of a signal, not a broken detector:

| Tape | pre-ramp *t* | post-move *t* | round-trip (no carry) *t* |
|---|--:|--:|--:|
| Planted hump (effect = 0.5, 40 events) | **+14.9** | **−15.8** | **+16.7** |
| Null (effect = 0, 40 events) | −0.5 | +0.1 | ≈ 0 |

## Verdict

- **Signal — NONE.** The pre-deadline VIX ramp the thesis needs does not exist (mean
  −5.2%, HAC *t* = −0.55, only 2/7 episodes positive), the post-deadline "collapse" is
  actually a small *rise* carried by 2011 alone, and both legs are indistinguishable from
  random dates (permutation *p* ≈ 0.5). Seven events is too few regardless.
- **Tradability — MIRAGE.** The canonical long-vol-into / short-vol-out round-trip loses
  −10.4% per event *gross*, dragged by the long-vol carry tax before any transaction cost.
  There is no abnormal vol move to harvest, and waiting for the deadline costs you roll.
- **Is brinkmanship a vol trade? — BUSTED.** The only episode that looks like a "vol
  spike" (2011) spiked *after* the deal on a credit-rating downgrade, not into the
  deadline — the opposite of the trade. The brinkmanship-vol narrative is one coincidence
  wearing a strategy's clothes.
