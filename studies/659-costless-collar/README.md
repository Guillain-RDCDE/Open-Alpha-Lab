# Study 659 — Costless-Collar 🪢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the floor and cap actually bite, and does the trade-off net out? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | Both legs are mechanically real: the floor cushions crashes **+2.68 pts/event (*t* = +6.39)**, the cap costs bull months **−1.83 pts/event (*t* = −7.67)**. Which one wins is regime-dependent — with 2008/2020 in the sample the net is a wash (*t* = −0.34); **excluding just those two named windows it flips to a certified drag (Newey-West *t* = −2.10)**. |
| **Tradability** — does the "free lunch" survive real-world stress? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Break-even cost is a razor-thin **3.84 bps/leg**; by 15 bps the drag is decisive (*t* = −3.25). The entire full-sample "win" is concentrated in **22 of 397 months** — a favorable realized-crash-timing draw, not a durable edge. |
| **"Truly free protection?"** | ![Busted](https://img.shields.io/badge/Free_protection%3F-Busted-8b949e?style=flat-square) | The cap's cost is real and mechanically guaranteed every time it binds — you always pay, whether or not a crash shows up to justify it. "Costless" describes the option premium at inception, not the trade. |

> **In one sentence:** a stylized SPY collar (own the index, 5% OTM put, call struck to make
> the premiums net to zero) really does cushion the 2008 and 2020 drawdowns by 9–19 points —
> but it pays for that with a statistically real, mechanically guaranteed cap on the upside,
> and once you strip out those two crash windows (or use realistic option-roll costs above a
> ~4 bps/leg break-even) the "free" lunch turns into a certified loss to buy & hold.

## What we tested

There is no historical SPY option chain on yfinance, so we build a **stylized monthly
collar**: each month the put is struck 5% out of the money and the call strike is solved
(Black-Scholes, priced off trailing realized volatility — our proxy for implied vol, since
there's no live chain) so its premium exactly offsets the put's — "costless" by construction.
The realized SPY return is then clipped to [−5%, that month's modeled cap], net of a 2-leg
monthly roll cost. We test whether the floor's crash cushion and the cap's upside cost are
each statistically real on the tape (they are, both ways), whether the net effect survives
excluding the two crash windows the claim itself invokes, and how thin the cost break-even
really is. **Dedup:** siblings [617-crash-insurance-cost](../617-crash-insurance-cost/) (a
naked put, no financing leg), [658-put-write-premium](../658-put-write-premium/) (the
mirror-image short-put trade), [337-covered-call-etf](../337-covered-call-etf/) (the covered
call half of this structure, with no protective put) and
[99-safety-net](../99-safety-net/) (a trailing-stop hedge, no options at all) each test a
different piece — this study is the one that puts the put *and* the call *and* the
"self-financing" framing together. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a collar actually is, why "costless" only ever describes the *day you put it on*, and what the marketing leaves out |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Black-Scholes cap solver, the floor/cap split tests, the cost-sensitivity and break-even sweep, the ex-crash-window robustness check, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`costless_collar/`](costless_collar/). A stylized options model, not a live option
chain — every approximation is named in [docs/references.md](docs/references.md). SPY is an
index ETF (no survivorship on the Signal axis). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
