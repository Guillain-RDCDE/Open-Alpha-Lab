# Study 888 — CLO AAA Carry 🔒

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the AAA-CLO pickup a *real* risk-adjusted edge? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | JAAA's **excess-of-cash carry is +1.38%/yr on 1.63% vol → Sharpe +0.84** (HAC *t* = **+2.33**, block-bootstrap CI **[+0.09, +1.72]** clear of zero) — the **top** of the excess-vs-excess Sharpe race, *above the un-tranched leveraged loans it is carved from* (BKLN +0.24, −24% drawdown vs JAAA's −2.6%), so the seniority/tranching earns real keep. **ICLO independently confirms** (+0.88, *t* = +3.08). Duration alternatives (LQD −0.11, IEF −0.33) have negative excess Sharpe over the 2022-crash window. *Caveats named: the carry is **regime-bound** (ZIRP era flat at +0.00%/yr — all of it is the high-rate era), the CI only just clears zero, and ~5.7y spans a single, stress-free credit cycle (the Mar-2020 AAA-CLO mark-down predates JAAA).* |
| **Tradability** — is it bankable net of costs? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The harvest survives costs & capacity trivially (buy-and-hold, ER already in the NAV, tight spreads on a >$20bn fund → **net +1.35%/yr, Sharpe +0.83**). What makes it *Fragile* not *Investable*: the whole ~1.4%/yr **is** the premium for a tail (illiquidity / senior-CLO mark-down) the calm sample never charged — a single crisis (AAA CLOs fell ~5-10% in Mar-2020) erases years of carry, so the realized −2.6% drawdown and the Sharpe flatter the true edge. |

> **In one sentence:** the senior AAA slice of a CLO really does pay a steady,
> low-vol pickup over cash and over the loans it's built from (Sharpe +0.84, HAC *t* = +2.33,
> ICLO confirms) — a genuine structural-complexity carry — but it is thin, regime-bound, and
> measured on a stress-free ~5.7y sample, so it is **real yet fragile**, not free money.

## What we tested

Since ~2020 the **AAA tranche of a CLO** — the senior, first-loss-protected slice of a pool of
leveraged loans — trades in a liquid, floating-rate ETF (**JAAA**, 2020-10; **ICLO**, 2022-12).
The pitch: harvest SOFR + a structural-complexity spread with ~no duration and ~no default risk.
We test that carry **excess-of-cash** (minus **BIL**) against IG corporates (**LQD**), 7-10y
Treasuries (**IEF**), and — the sharp control — the *un-tranched, below-IG* loan collateral
itself (**BKLN**): a yfinance total-return tape, 2020-01-02 → 2026-06-30. Per leg we compute the
annualised excess, vol, Sharpe with a block-bootstrap 95% CI, a Newey-West *t* on the daily
excess, the drawdown, a ZIRP-vs-high-rate era cut, a costed buy-and-hold harvest + long/short
isolation trade, and a 12-seed synthetic control. **Short history is named on the Signal axis**
(~5.7y, one rate cycle, no CLO stress event → Sharpes are an upper bound). **Dedup:**
[614-clo-equity-yield](../614-clo-equity-yield/) is the risky **first-loss EQUITY tranche** at
the *bottom* of the same stack (this is the senior **top**); [340-bank-loans](../340-bank-loans/)
is the **un-tranched** loan pool (here BKLN is the *control* the AAA slice must beat risk-adjusted);
[796-corporate-bond-low-risk](../796-corporate-bond-low-risk/) is a low-risk anomaly *within*
corporates, not the securitised wrapper; [885-ultra-short-credit-pickup](../885-ultra-short-credit-pickup/)
is the ultra-short *corporate* pickup, not CLO tranches. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the safest slice still pays a spread — and the tell: it beats the *raw loans* it's built from, risk-adjusted |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-Sharpe race with bootstrap CIs, the HAC *t*, the JAAA-vs-benchmark head-to-heads, the era cut, the cost math, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`clo_aaa/`](clo_aaa/). Real tape via yfinance (total-return), cached under `_cache/`.
Short-history / no-stress caveat travels with every number (Signal axis).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
