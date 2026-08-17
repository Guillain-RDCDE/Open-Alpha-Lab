# Study 954 — High Yield in Disguise 🎭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-d68910?style=flat-square) | Two questions, two answers. The *costume* claim is **refuted**: the held-out blend explains only **R² = 0.47** daily (0.52–0.59 weekly-to-quarterly) and leaves ~7 pp/yr of tracking error — high yield is a genuinely distinct risk. The *compensation* claim points one way in every single cut (gap −0.25 to −0.39 across four estimation windows, four Treasury maturities, both eras, all three funds, never the other sign) but only **just** clears the bar: HAC *t* = **−2.07**, bootstrap CI [−0.81, −0.04], while JNK (−1.64), USHY (−1.24), the 2008–2016 era (−1.19) and the SHY (−1.93) / TLT (−1.92) duration legs all miss it. Survivorship: three still-listed funds, dead high-yield ETFs absent — which flatters the *fund* side, so the negative finding is conservative. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-d68910?style=flat-square) | The swap is cheap and real — 0.34 ×NAV/yr turnover, unchanged out to a punitive 25 bps, no leverage or borrow, **+2.5 pp/yr of lived CAGR** (7.7% vs 5.2%) and **−11.7 pp of worst drawdown** over eighteen years — but it is a substitution, not an alpha, and it swaps credit risk for duration risk: in the 2022 rate shock the blend lost **4.7 pp more** than the fund it replaced. |

> **In one sentence:** High yield is *not* levered equity in a bond costume — a held-out `w·SPY + (1−w)·IEF` blend replicates less than half of it — yet on eighteen years of tape that genuinely distinct credit risk paid **nothing** for itself: at matched volatility the 45/55 blend beat HYG by ~3.7 pp/yr with a shallower worst loss, on a *t* of −2.07 that is unanimous in direction and marginal in size.

## What we tested

Fit **HYG** (and JNK, USHY) to a fully funded **`w · SPY + (1 − w) · IEF`** blend by
trailing-252-day constrained regression, freeze `w` at each month-end and apply it to the
*following* month (the single execution lag), charge the blend 2 bps one-way on turnover
and a borrow spread on any short leg (the fitted `w` stayed inside [0.27, 0.63], so borrow
is inert), and race both arms **excess-of-cash** (BIL) over the held-out window
2008-06-02 → 2026-06-30 (total-return closes, `auto_adjust=True`). Then the practical
question: at the same realised volatility (an *ex-post* match — a test statistic, not a
tradable path), which paid more, and which hurt worse in 2008, 2020 and 2022? Vol-matched
HAC *t*, block-bootstrap CIs, an era cut, a cost sweep, an estimation-window sweep and a
**duration-leg sweep** (SHY / IEI / IEF / TLT — Treasuries only, so no credit ever enters
the benchmark). **Dedup:** distinct from **115-credit-spreads** (spread as a macro
*timing signal*), **832-high-yield-credit-momentum** (a timing rule *inside* high yield),
**865-credit-equity-lead-lag** (does credit move *first*), **885 / 887 / 907** (sleeve races
*within* credit), **951-crossover-credit-bbb-bb** (where on the ladder to sit), and
**953-convertible-replication** (same machinery, converts and an IG-credit leg) — here the
benchmark deliberately contains **no credit at all**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a junk bond really is, the 45/55 recipe, why the copy still won, and the one year it did not |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the constrained held-out regression, R² by horizon, vol-matched HAC *t*, bootstrap CIs, era cut, crisis table, cost / window / duration-leg sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hy_replication/`](hy_replication/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
