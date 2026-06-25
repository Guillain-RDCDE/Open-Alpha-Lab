# Study 500 — Polarity-Flip 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does broken resistance bounce? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The "buy the retest of broken resistance" rule **does** beat a drift-matched **random-entry** baseline — but **only at the 5-day horizon**: Welch *t* = **+2.05** (*p* = 0.040), with a positive retest−random delta in **all 5 names** (+15 / +29 / +63 / +8.5 / +28.5 bps). The edge **decays** by 10/20/60 days (Welch *t* = 1.44 / 1.23 / 1.47). The big one-sample *t*'s (20d **+3.68**, 60d **+6.34**) are mostly beta. A real but short-lived bounce. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The 5-day edge is real but **thin** (Δ ≈ +29 bps), **gone by ~10 days**, exposed to slippage on a ±1% retest trigger, and at 20 days leans on just QQQ/GLD; costs trim it. Tradable only as a minor short-horizon tilt — nothing to scale. |
| **"Does broken resistance hold as support?"** | ![Mixed](https://img.shields.io/badge/Holds_as_support%3F-Mixed-dab617?style=flat-square) | At the **first** retest, over the **next few days**, yes — a measurable bounce beyond drift (5d *p* = 0.040). But it's brief, and the level-scramble placebo doesn't clear 0.05 (*p* = **0.122**), so we can't pin the bounce decisively on *these specific old levels* vs generic post-pullback mean reversion. Partial confirmation. |

> **In one sentence:** Unlike most chart folklore, role reversal isn't pure mirage — encode it mechanically (confirmed swing-high levels, break-then-first-retest, no eyeballing) and fire it 598 times across 5 indices over 21 years, and the retest genuinely beats buying on random days **at 5 days** (Welch *t* = +2.05, positive in all 5 names); but the bounce **fades within two weeks** and the level-scramble placebo (*p* = 0.12) can't prove it's the specific old level rather than generic short-term reversion — a real, fragile, short-lived effect.

## What we tested

We encode the tightest mechanical version a proponent would accept. Resistance levels are **confirmed swing-high fractals** (a local maximum with *k* = 10 strictly-lower bars each side, usable only 10 bars later — no look-ahead); a level becomes **broken** the first time the close prints +0.5% above it; a long fires on the **first pullback** back into a ±1% band around the broken level (the polarity-flip retest), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **retest vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **scrambled-level placebo** that permutes which price is the level while keeping the marginal. Tradability charges costs on every retest. A deterministic synthetic control with a *planted* role-reversal bounce proves the detector is live (edge 0 → *t* = −0.20; planted flip → *t* = +5.56), so the borderline real-tape result is a genuine measurement.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what role reversal is, why a dip-buy on a rising market always looks good, the retest-vs-random race (and the real 5-day edge), and the level scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical break-then-retest levels, one-sample HAC *t* vs the beta trap, the random-entry Welch test (5d clears 2, then decays), the all-5-names 5-day coherence, the scrambled-level placebo, per-ticker deltas, costs, and a synthetic planted-flip control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`polarity_flip/`](polarity_flip/). Levels are confirmed swing-high fractals (k = 10) with a 10-bar confirmation lag; break = close +0.5% above, retest = first close within ±1%; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument level study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
