# Study 374 — Vol-of-Vol 🌀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the vol-of-vol time equity risk? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Raw high-VVIX days *do* precede slightly **higher** mean returns (Welch *t* = **2.35** at 1m) and deeper drawdowns — a real but **direction-confused** pattern (it flags turbulence, not down-moves). But the claim is *"better than the VIX"*, and that fails the decisive test: once VIX is in the regression, VVIX's HAC *t* drops to **1.57 / 1.12**, and the raw *t* is threshold-fragile (**0.56** at the 70th pct). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A long/flat "cash when vol-of-vol is high" timer nets a Sharpe of **0.65** — a statistical tie with buy-and-hold's **0.63**. It buys vol reduction any cash blend would, at the cost of return. No incremental signal over the VIX ⇒ no NAV-scale edge. |
| **Better than the VIX?** | ![Busted](https://img.shields.io/badge/Better_than_the_VIX%3F-Busted-8b949e?style=flat-square) | The selling point is *incremental* timing power. The HAC regression says VVIX adds nothing robust once VIX is in: the raw VVIX *t* of **2.28** is mostly the **0.41**-correlated VIX leaking through. Vol-of-vol is a redundant cousin of vol, not a sharper risk clock. |

> **In one sentence:** VVIX, the volatility of the VIX, looks like a risk timer in isolation (high-VVIX days even run a positive Welch *t*), but it points the wrong way for the "spike = sell" story, is fragile to the threshold, and — decisively — adds **nothing robust over the VIX itself** once both are in one HAC regression (VVIX *t* falls from 2.28 to 1.57), so it's real-ish as a turbulence flag, weak as an edge, and a redundant cousin of the vol you already track for free.

## What we tested

VVIX is the CBOE "volatility of the VIX" — the VIX construction applied to VIX *options*, a model-free read of the expected vol *of* vol. The believers' claim is that a VVIX spike prices a tail the VIX level hasn't shown yet, so it should time equity risk **better than the VIX**. We fetch `^VVIX`, `^VIX` and `SPY` (yfinance, 2007→2026, 4,887 days), build a point-in-time "high vol-of-vol" state (VVIX above its trailing 252-day 80th percentile), and measure forward SPY returns and drawdowns vs the base rate. The decisive test is **incremental**: a Newey-West (HAC) regression of forward returns on standardized VIX **and** VVIX — does vol-of-vol add anything once the VIX is controlled? A deterministic synthetic control plants an edge **only** in the VIX-orthogonal part of VVIX, proving the engine isolates incremental signal rather than VIX bleeding through.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "vol of vol" even is, why high-VVIX days look scary but precede *rebounds*, and why "better than the VIX" is the only question that matters — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the high-VVIX state, conditional vs unconditional forward returns/drawdowns, the decisive VIX+VVIX HAC regression, a block placebo null, a costed long/flat timer, and a synthetic incremental-signal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vol_of_vol/`](vol_of_vol/). VVIX/VIX are index levels; SPY is auto-adjusted (total-return-ish). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
