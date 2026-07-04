# Study 640 — Gold-Overnight 🌙🥇

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does gold really trade in its sleep? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | GLD's overnight-minus-intraday gap is **+4.08 bps/day** at HAC ***t* = 2.94** (sign-flip *p* = 0.003), confirmed on IAU at **+5.39 bps/day, HAC *t* = 3.83**; over 21.6 years the overnight sleeve grew **8.63×** while the intraday sleeve went **nowhere (0.96×)**. Caveats said out loud: gold's gap is ~2× SPY's but **not significantly bigger** than the market-wide clock effect (Welch *t* = 1.00 vs the placebo), and GLD's pre-2015 sub-sample alone sits at *t* = 1.35. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even **frictionless**, holding GLD only overnight earns +10.51%/yr vs **+10.31%/yr for just holding it** — the intraday leg you'd dodge is a statistical zero (*t* = 0.22). With 2 trades/day the overlay trails buy & hold at **every** cost level in **every** sub-period (+7.76% net at an unrealistic 0.5 bps; −0.08% at 2 bps). |
| **"Ever harvestable net of spreads?"** | ![Busted](https://img.shields.io/badge/Ever_harvestable%3F-Busted-8b949e?style=flat-square) | Break-even one-way cost ≤ ~0.18 bps (IAU) and *negative* on GLD — below any spread GLD ever traded at (≈2 bps in 2005, ≈0.3 bps today). Bonus: the **2015 LBMA fix reform didn't dent it** (gap +3.07 → +5.01 bps/d, change Welch *t* = −0.65) — the fix-manipulation mechanism fails; the boring clock-and-venue story stands. |

> **In one sentence:** gold genuinely earned its entire two-decade return while US exchanges were shut (HAC *t* ≈ 3–4 on both GLD and IAU) — but the anomaly is a description of *when* the return arrives, not a trade: there is nothing to dodge intraday, two spreads a day bury it, its edge over the market-wide overnight effect of [study 01](../01-overnight-anomaly/) is statistically unproven, and the 2015 fix reform it was blamed on left it untouched.

## What we tested

The folklore — from Adrian Douglas's GATA chart to Caminschi-Heaney's leaky London PM fix — says gold rises **overnight** (Asia + the London AM fix, while NYSE Arca is shut) and stalls **during London/NY hours**. We split every GLD session 2004→2026 into overnight (close → open) and intraday (open → close) legs on total-return opens/closes, test the gap with a Newey-West *t* on the paired daily difference plus a 20,000-draw sign-flip placebo, **confirm on IAU** (different sponsor, same metal), and judge the size against the **SPY placebo** — [01-overnight-anomaly](../01-overnight-anomaly/) already showed equities do this, so gold must beat that bar, not a vacuum. The **2015-03-20 LBMA fix reform** gives an externally-dated natural experiment on the claimed mechanism (Welch *t* on the change in gap). Tradability charges a buy-MOC/sell-MOO overlay 2 one-way trades/day at 0.5–5 bps vs buy & hold. A deterministic synthetic world with a planted overnight drift proves the machinery (never cited for the stamp). As-of **2026-06-30**, fingerprint `648ab68f37f2`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the sleeping-gold chart drawn honestly, why the *entire* gain is overnight, why you still can't trade it, and what the 2015 fix reform proves — in plain words |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the paired night/day difference, the sign-flip placebo, IAU confirmation, the SPY-placebo Welch race, the pre/post-reform contrast with its own test, costs × 2-legs-a-day, and the planted-drift synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gold_overnight/`](gold_overnight/). The signal is the overnight-minus-intraday session gap on adjusted opens/closes; the myth-checks are the SPY placebo, the 2015 LBMA-reform break, and the 2-trades-a-day harvest arithmetic. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
