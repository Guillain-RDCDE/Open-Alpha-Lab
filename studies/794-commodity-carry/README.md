# Study 794 — Commodity Carry 🛢️📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do backwardated commodities out-earn contangoed ones, cross-sectionally? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Pooled cross-sectional carry→forward-return Newey-West(6) *t* = **−0.95** (slope −0.0145, corr −0.055) over **205 two-name months** — not significant and, if anything, wrong-signed. Backwardated months average **−1.01%/mo** forward vs **−1.20%/mo** in contango (Welch *t* = **+0.13**), and the backwardated hit rate is **43.8%** (46/105, Wilson [34.7%, 53.4%]) — below a coin flip. Two energy curves is the thinnest cross-section possible: this can neither certify nor refute the broad-universe factor. |
| **Tradability** — can you harvest a two-name carry long-short? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The monthly two-name long-short loses money **before** costs — gross **−9.46%/yr** (NW *t* = −1.48, Sharpe −0.35) — and net of 5 bps + 100 bps/yr short borrow it bleeds **−12.86%/yr** (NW *t* = **−2.01**, Sharpe −0.48, worst month **−31.3%**); at 10 bps it's **−15.26%/yr** (*t* = −2.38). Two crash-prone energy legs are not a harvestable premium. |
| **Is the roll mechanism itself real?** | ![Confirmed](https://img.shields.io/badge/Roll_mechanism-Confirmed-8b949e?style=flat-square) | Yes, cleanly, on the investable tape: the **USO-minus-USL** front-vs-ladder gap regressed on WTI carry gives NW(6) *t* = **+2.23** (slope +0.0409), and the gap is less negative in backwardation (−0.33%/mo) than in contango (−0.76%/mo). Backwardation genuinely makes the front fund out-roll its 12-month ladder — the mechanism was never the weak link; certifying a *tradable cross-sectional premium* on two names is. |

> **In one sentence:** the roll mechanism is real — backwardation makes the front-month
> fund out-roll its laddered twin (USO-USL vs carry, NW *t* = **+2.23**) — but on a
> two-name energy proxy the cross-sectional carry→return premium doesn't clear *t* ≥ 2
> (NW *t* = **−0.95**, if anything wrong-signed), and a costed, borrow-charged long-short
> loses money gross (**−9.46%/yr**) and net (**−12.86%/yr**, *t* = −2.01), so the honest
> read is a **genuine mechanism weighed on an under-powered, un-tradable proxy**.

## What we tested

The best-documented premium in commodities: *own the backwardated curves (positive roll
yield), avoid the contangoed ones, and collect* (Gorton-Rouwenhorst 2006; Erb-Harvey
2006; Koijen et al. 2018). Because clean, broad historical commodity term structure is
**not freely available**, we test it as an honest **two-name energy proxy**: the ex-ante
carry signal is read from the *real* EIA futures term structure (WTI `RCLC1-2`, Henry Hub
gas `RNGC1-2`, annualized roll yield `ln(C1/C2)·12`), and the outcome is the *next*
month's return of the investable front-month ETF (USO for crude, UNG for gas) — one
documented execution lag, zero look-ahead. The workhorse is a **cross-sectionally
demeaned carry→forward-return** Newey-West regression; a backwardation-vs-contango Welch
split and Wilson hit rate cross-check it; the **USO-minus-USL** gap on the same crude
regresses realized roll on carry as a mechanism check; and a costed, borrow-charged
two-name long-short grades tradability. A seeded synthetic control with a tunable planted
premium proves the machinery is unbiased. **Proxy caveat, stated plainly:** two names is
the thinnest cross-section possible — every cross-sectional number here is under-powered
by construction and can neither confirm nor refute the broad-universe factor.
**Dedup:** siblings [35-contango](../35-contango/) (grades the *realized roll drag* via
the laddered-minus-front ETF gap, not the ex-ante curve signal),
[380-curve-roll-down](../380-curve-roll-down/) (single-asset roll-down in *rates*),
[660-carry-everywhere](../660-carry-everywhere/) (the *multi-asset* carry blend, commodity
is one of four legs) and [661-uso-roll-decay](../661-uso-roll-decay/) (the *single-fund*
USO-vs-spot decay) — this study isolates the **cross-sectional commodity** carry sort and
reads carry from the actual futures curve. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the curve slope is supposed to pay you, why the roll mechanism is genuinely real on the ETF tape, and why two energy curves still can't establish a tradable cross-sectional premium |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the cross-sectionally demeaned carry→return HAC regression, the backwardation/contango Welch split + Wilson hit rate, the USO-USL roll-mechanism regression, the costed + borrow-charged long-short, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`commodity_carry/`](commodity_carry/). Carry from the EIA futures term
structure; returns from investable energy ETFs. A **two-name energy proxy** for the
general cross-sectional commodity carry premium — under-powered by construction.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
