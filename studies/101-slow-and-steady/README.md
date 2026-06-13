# Study 101 — Slow-and-Steady 💧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does DCA beat a lump sum? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Across **8,168** rolling 12-month windows on SPY (total return), the **lump sum finishes richer 78.6%** of the time — more than Vanguard's ~2/3 — and the mean gap of **+5.6 cents/\$1** crushes the bar even after the overlap correction (HAC *t* = **+16.96**). DCA only wins when the market *falls* over the window. |
| **Tradability** — is DCA the *smarter* way to invest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The lump sum out-earns DCA by **5.6 cents on the dollar** on average; the only thing DCA "buys" is lower exposure to the equity premium it then forgoes. A lower stock weight gets you the same calmer ride with no opportunity cost. |
| **Lowers risk?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Real risk reduction: DCA's spread of outcomes is **little over half** the lump sum's (dispersion ratio **0.536**), and its worst window loses **−37.5%** vs **−49.3%**. But it's lower *exposure*, not better *timing*. |

> **In one sentence:** spreading your money in genuinely **lowers the dispersion** of where you land (half the spread, a shallower worst case), but it does **not** beat going all-in — the lump sum wins **~79%** of the time and **+5.6 cents on the dollar**, because every dollar that sits in cash "averaging in" is a dollar forgoing the returns markets pay on average.

## What we tested

The most-repeated piece of money advice, at full strength: *"don't invest a windfall all at once — dollar-cost average it in over a year. You'll buy at a lower average price, you'll lower your risk, and you'll beat going all-in."* We take it literally — for **every** rolling start date, race a \$1 **lump sum** (invested at *t0*) against \$1 of **DCA** (12 equal monthly tranches, cash earning **0%** while it waits — a choice that *helps* DCA's case), value both at the **same** end date, and tally the winner. We cite the [Vanguard 2012/2023 study](docs/references.md) (lump-sum wins ~2/3 of the time) and [Constantinides (1979)](docs/references.md) (DCA is provably sub-optimal). A deterministic synthetic tape with a single drift knob is the two-sided positive control: it banks **DCA** when the tape falls and **lump-sum** when it rises.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, the win-rate, the distribution of who-finishes-richer, why the smoother ride costs return, the one thing DCA really does buy |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | win-rate with a Wilson CI, the HAC *t* on the overlapping-window gap, the dispersion-ratio risk read, exposure-vs-timing decomposition |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`slow_and_steady/`](slow_and_steady/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
