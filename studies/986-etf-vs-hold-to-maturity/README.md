# Study 986 — The Rolling Ladder 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a constant-maturity bond fund converge to its starting yield the way a bond does? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | A bond's starting yield is its return; a fund's is a forecast. Over 203 rolling 7.5-year windows — a horizon equal to IEF's own duration, where immunisation theory says the error should have washed out — the realised annualised return missed the starting yield by **+3.46% on average with a standard deviation of 2.32%**, and only 21% of windows landed within a percentage point of what was on offer. The mechanism checks out: regressing the error on **−duration × Δyield ⁄ horizon** gives a slope of 13.34 with R² **91%**, so the shortfall is the refusal to mature and essentially nothing else. At one year the error's standard deviation is 6.4%; it falls to 1.9% at ten years — shrinking, but nothing like a bond's, which is zero at maturity by construction. |
| **Tradability** — does the difference cost or pay you, and over what horizon? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | In the controlled experiment — one rate path, a 10-year bond held to maturity against a 10-year constant-maturity fund, a 200 bp rise a year in — the fund fell 0.5% behind and took **5.6 years** to catch the bond, against a starting duration of 8.2. On a steadily trending rate path the fund's annualised return crossed back through its purchase yield after **11.1 years** — against Leibowitz, Bova & Kogelman's 2D − 1 = 15.4, and *below* it because the fund's duration shrinks as yields rise, so the cumulative price loss is smaller than D₀ × Δy. That is the practical content: the fund is not worse, it is *slower to be right*, and the delay is set by its duration. An investor whose horizon comfortably exceeds it is close to indifferent; one whose horizon is shorter is holding a rate bet they may not know they have taken. |

> **In one sentence:** A constant-maturity bond fund delivers its starting yield only after roughly 1.4 times its duration — 11.1 years in the clean experiment — and until then the gap is 6.4% a year of pure rate risk that a bond held to maturity simply does not have.

## What we tested

Buy a ten-year Treasury at 4% and hold it to maturity and you earn 4% a year,
whatever rates do in between: the bond pulls to par and its duration falls to zero on a
schedule. Buy a ten-year Treasury *fund* at 4% and you do not, because the fund sells each bond
as it ages out and buys a fresh one — it never matures, never pulls to par, and its duration
never falls. Everyone in fixed income knows this. Almost nobody can say **how big the difference
is, which way it runs, or when it resolves**.

This study answers all three. First a controlled experiment: one interest-rate path, a bond held
to maturity against a constant-maturity fund, so every difference between them is the roll and
nothing else. The fund falls behind immediately after a rate rise, then catches up — and the
**crossover horizon is its duration**, which is Redington's 1952 immunisation result arriving
from the other direction. Then the same question on the tape: for SHY, IEF, TLT and LQD, every
rolling window's starting yield against the annualised return that actually followed, with the
error regressed on **−duration × Δyield ⁄ horizon**. A slope near one and a high R² would mean
the entire gap between what a bond fund advertises and what it delivers is its refusal to
mature, with no second mystery to explain.
**Dedup:** distinct from **203-bond-etf-vs-index** (tracking error and premium/discount),
**419-duration-risk** (duration as a cross-sectional risk factor), **556-rising-rates-and-bonds**
(the directional question), **744-tips-vs-nominals** (a different instrument comparison) and
**662-cash-vs-bonds** (an allocation question that assumes the fund and the bond are the same
thing — which is exactly what this study takes apart).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a bond fund with a 4% yield will not give you 4%, how long the discrepancy lasts, and the one number that tells you when it resolves |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | bond arithmetic verified against its textbook identities, a controlled one-path experiment, crossover horizons across maturities and shocks, and the shortfall decomposed into duration times the rate change on the real tape |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ladder/`](ladder/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
