# Study 892 — Corporate-Bond Ladder 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a held-to-maturity ladder beat a constant-maturity fund? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Duration-matched, the Treasury ladder minus the AGG fund is **−0.01 %/yr at HAC *t* = −0.02**; the difference-Sharpe bootstrap CI **[−0.48, +0.46]** straddles zero and the sign **flips across eras** (+0.67 / −0.60 / −0.53 %/yr, all \|*t*\| < 1.1). The naive equal-weight ladder actually **underperforms** (excess-Sharpe **0.28 vs 0.39**) purely on 1.5y of extra duration. "Held-to-maturity beats the fund" is an **accounting illusion** for default-free bonds — pull-to-par is the exact reversal of the fund's mark-to-market loss. |
| **Tradability** — can you bank it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is **no gross edge** to harvest, and the ETF ladder pays annual roll/rebalance costs (**0.9–3.0 bps/yr**) that the one-ticker buy-and-hold fund does not, so net it is **strictly behind** (−0.02 to −0.04 %/yr, HAC *t* ≈ 0). Even the 2022 "win" (+0.60 pp) was a credit-spread dodge fully reversed in 2023 (−1.75 pp). The ladder's appeal is **behavioral**, not risk-adjusted return. |

> **In one sentence:** once you match duration, a Treasury ETF ladder and the AGG/BND
> aggregate fund are a **statistical dead heat** (−0.01 %/yr, HAC *t* = −0.02) — the famous
> "ladder holds to par so it beats the forced-selling fund" story is an accounting illusion
> for default-free bonds, and every apparent ladder edge (or the naive equal-weight ladder's
> *underperformance*) is a **duration + credit-composition bet in disguise**.

## What we tested

We race a duration-staggered Treasury **ladder** (SHY/IEI/IEF/TLT, annually rolled) against
the **AGG**/**BND** constant-maturity **funds** on yfinance total-return closes (2007-06 →
2026-06, 229 months), excess of T-bill cash (**BIL**). Two ladders: the naive **equal-weight**
basket (duration 7.5y) and a **duration-matched** ladder (6.0y, tuned to AGG — the fair
control). Inference is an excess-of-cash Sharpe race with **circular-block-bootstrap** CIs,
**Newey-West HAC *t*** (6 lags) on the ladder-minus-fund monthly diff, a three-era cut, a
2022-rate-shock calendar-year row, and a costed net that charges the ladder its annual roll
while leaving the buy-and-hold fund free. A deterministic synthetic world with a **planted,
tunable ladder premium** proves the detector recovers the knob and that a zero-premium null
cannot fire. Honest caveats stated: the ETF proxy is itself *constant-maturity* (a real
ladder needs defined-maturity iBonds/BulletShares rungs), AGG carries credit/MBS that pure
Treasuries don't, and the joint window is a ~19-year survivor tape. **Dedup** — distinct from
[59-downhill](../59-downhill/) (directional duration *timing*, not two static structures),
[380-curve-roll-down](../380-curve-roll-down/) (roll-down accrues to *both* a ladder and a
fund, so it is not the differential here), [884-convexity-barbell](../884-convexity-barbell/)
(a second-order *convexity* trade, not a first-order duration/credit gap), and
[625-starting-yield](../625-starting-yield/) (starting yield as a mechanical *identity*; we
ask the orthogonal question — same yield/duration, does the *structure* add anything?).
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why holding to par doesn't beat marking to market for a default-free bond, why the naive ladder *loses*, and what actually happened in 2022 — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race with bootstrap CIs, the Newey-West diff *t*, the era cut, the 2022 calendar-year stress row, the costed net, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bond_ladder/`](bond_ladder/). The race compares a fixed-weight Treasury-ETF basket
to a one-ticker aggregate fund on monthly total returns; effective durations are coarse
hardcoded fact-sheet values used only to duration-match. **Not investment advice** — research
& education. See [LICENSE](../../LICENSE).*
