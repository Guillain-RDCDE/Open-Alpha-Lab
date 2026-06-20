# Results — Study 308 (Cocoa-Squeeze) on the real cocoa tape

*The 2024 cocoa parabola, dissected. Front-month cocoa futures (`CC=F`, Yahoo, daily,
**price-only** continuous roll — not a total-return index) from 2000-01-03 to the last
full month. We locate the blow-off, run the two folk reactions (ride the momentum / fade
the crack) with one execution lag, HAC `t`, block-bootstrap CIs, costs one-way × turnover
× NAV, and shorts paying borrow. The deterministic synthetic blow-off is the positive
control. As-of **2026-05-29** (June 2026 partial month dropped); match the fingerprint to
confirm you hold the same tape.*

## Data stamp

| Ticker | Window | Days | Fingerprint |
|---|---|--:|---|
| CC=F (cocoa front-month, price-only roll) | 2000-01-03 → 2026-05-29 | 6,622 | `077d24a7d94b` |

## The blow-off itself — anatomy of the parabola

| Quantity | Value |
|---|--:|
| Pre-blowoff trough | 2024-01-08, ~4,094 $/t |
| All-time-high peak | **2024-12-18, ~12,565 $/t** |
| Run-up multiple (trough → peak) | **3.1×** |
| Forward drawdown from the peak (worst) | **−77.7%** |
| Forward return +20d / +60d / +250d from peak | −11.1% / −36.2% / −52.3% |

The famous April-2024 spike was only a *local* peak; cocoa ran higher into December 2024,
topping near $12,600/t, then gave back roughly three-quarters of the run. A textbook
blow-off-and-crack.

## Did "ride the parabola" (momentum) pay? — No

Long while the price is stretched ≥1σ above its 100-day trend *and* still rising; one
execution lag; full real tape.

| Cost (one-way bps) | ann. return | HAC *t* |
|---|--:|--:|
| 0 (gross) | +0.1%/yr | **+0.08** |
| 5 | −0.1%/yr | −0.05 |
| 10 | −0.2%/yr | −0.18 |
| 20 | −0.5%/yr | −0.44 |

- Even **gross**, the momentum timer's HAC *t* is **+0.08** — indistinguishable from zero.
- Restricting to the 2024 blow-off window itself: +12.7%/yr but **HAC *t* = +0.91** on 33
  active days — a single lucky episode, not a robust edge. You whipsaw in and out of a
  parabola that gaps both ways.

## Did "fade the blow-off" (short the crack) pay? — No, it *lost*

Short for 20 days once price set a multi-month high and then fell ≥7% from its recent max
(the disciplined "don't stand in front of it, short the roll-over" version). Shorts pay a
300 bps/yr borrow.

| | active days | mean (bps/day) | 95% block-bootstrap CI (bps/day) | ann. return | HAC *t* |
|---|--:|--:|--:|--:|--:|
| Crack-fade short | 111 | **−61.5** | **[−121.8, −12.6]** | **−2.6%/yr** | **−1.99** |

- The fade **lost money** — its block-bootstrap CI of the mean daily return is **entirely
  below zero**. Shorting a cracking parabola is the widow-maker the folklore warns about:
  the sharp bear-market rallies on the way down stop you out.
- A naive "short when stretch z > 2" version is even worse (it shorts a still-rising
  market). There is no robust tradable short here.

## The single-event problem — why the Signal is `NONE`

The 2024 cocoa parabola is **one event**. There is no cross-section: a single asset, a
single blow-off. The desk's inference bar requires a robust HAC *t* ≥ 2 **on the real
tape** for a `REAL` stamp. Here the real tape delivers *t* = +0.08 (momentum) and *t* =
−1.99 (fade, the wrong sign and not robust). One spectacular chart is not a measurement.

## Synthetic positive control — the engine works, the market didn't cooperate

On a deterministic synthetic tape with a *planted* blow-off (a ~2× parabola that then mean-
reverts), the same momentum engine recovers a clear edge — and on a pure random walk it
does not:

| Synthetic tape | momentum HAC *t* | momentum ann. return |
|---|--:|--:|
| Planted blow-off (bubble = 1) | **+3.76** | +11.8%/yr |
| Pure random walk (bubble = 0, the null) | +1.52 | ≈ 0 |

The harness *can* detect a tradable blow-off when one is planted. That it finds nothing
robust on cocoa is a statement about cocoa's *one* parabola, not a broken detector. **A
synthetic control proves machinery, never markets** — it can never back a Signal stamp.

## Verdict

- **Signal — NONE.** On the real cocoa tape neither leg clears the bar: ride-the-momentum
  HAC *t* = +0.08, fade-the-crack *t* = −1.99 (losing, CI entirely negative). With a single
  blow-off there is no cross-section to certify anything; the synthetic control proves only
  that the engine works.
- **Tradability — MIRAGE.** Momentum is a coin flip gross and negative net of any cost;
  the fade reliably *loses* money once shorts pay borrow and eat the bear rallies. Nothing
  here survives contact with costs, and the one event cannot be sized or repeated.
- **A repeatable cocoa edge? — NOT SUPPORTED.** The −77.7% give-back *looks* like an
  obvious "should have shorted it" in hindsight, but the timed short lost money and the
  long timer earned nothing. The parabola was a once-in-a-generation supply shock
  (West-African crop failure), not a recurring, harvestable pattern.
