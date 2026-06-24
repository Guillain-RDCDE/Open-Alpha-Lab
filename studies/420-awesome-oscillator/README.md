# Study 420 — Awesome Oscillator

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the timing real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The AO long/flat rule's **excess Sharpe (+0.185)** is *below* buy-and-hold (**+0.337**); the difference HAC *t* = **−1.91** has the wrong sign. A rotation permutation places the edge at *p* = **0.92** (most random shifts beat it). No certifiable signal. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Loses the race even gross. Its lower drawdown (**−39%** vs −64%) is matched by a random coin out of the market as often (Sharpe **+0.211** ≈ AO **+0.185**) — exposure reduction, not skill — and is paid for with lower return. |
| **"Beats MACD?"** | ![Busted](https://img.shields.io/badge/Beats_MACD%3F-Busted-8b949e?style=flat-square) | MACD's excess Sharpe (**+0.250**) is *higher* than the AO's; the AO−MACD difference *t* = **−0.88**. The midpoint input buys nothing over the close-based MACD — if anything MACD is marginally better. |

> **In one sentence:** Bill Williams' Awesome Oscillator — pitched as a "leading" momentum gauge that beats the MACD — turned into a long/flat SPY timing rule does not beat buy-and-hold (excess Sharpe +0.185 vs +0.337, difference *t* = −1.91), does not beat a plain MACD (+0.250, difference *t* = −0.88), and its timing is indistinguishable from a coin out of the market just as often (permutation *p* = 0.92); the smaller drawdown it shows in screenshots is pure exposure reduction, not skill.

## What we tested

A TradingView staple: *"AO = 5-period SMA minus 34-period SMA of the bar midpoint (high+low)/2; be long when the histogram is above zero, step aside below — and because it reads the midpoint, it leads the lagging MACD."* We take the steelman literally: a daily **long/flat** rule on SPY total-return OHLC (1993–2026, n = 8,400), hold when AO > 0 else T-bills, entered one day after the signal (one execution lag). We race its **excess-of-cash** Sharpe against buy-and-hold and against the *identical* rule run on the **MACD(12/26)** line — so "beats MACD" is adjudicated, not assumed — plus a matched-exposure **random-timing** control. Inference is a HAC (Newey-West) *t* on each excess return difference; a 2,000-draw **rotation permutation** placebo guards against reading noise; costs are 2 bps one-way × NAV (swept 0–10 bps, long-only no borrow). A deterministic **synthetic two-regime tape** with a *planted* bear market confirms the harness banks a real edge when one exists (*t* = +2.57) and finds none when it doesn't — proving the null on SPY is informative, not a power failure.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the AO is, why ducking a crash on a chart is persuasive but not skill, the fair race against MACD and a random coin, why a smaller drawdown isn't an edge — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | AO(5/34) vs MACD(12/26) long/flat on excess returns, HAC difference *t*-stats (AO vs BH / MACD / random), a rotation permutation placebo, sub-period decay, cost sweep, and the synthetic planted-bear power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`awesome_oscillator/`](awesome_oscillator/). Races are **excess-of-cash vs excess-of-cash**; SPY is split/dividend-adjusted total return; cash leg proxied at 4%/yr flat. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
