# Study 489 — Chaikin Oscillator 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does A/D momentum forecast price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the cross above zero" rule does **not** beat a drift-matched **random-entry** baseline: cross − random = **−5.9 / −22.3 / −21.9 / −69.8 bps** at 5/10/20/60 days, and the cross-vs-random Welch *t* is **never positive** — at 60d it is *significantly negative* (**−2.14**, *p* = 0.032: the cross actively underperforms a dart). The big one-sample *t*'s (20d **+4.88**, 60d **+5.81**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"A/D momentum leads price"?** | ![Busted](https://img.shields.io/badge/A%2FD_leads_price%3F-Busted-8b949e?style=flat-square) | Scramble the accumulation readings into nonsense (shuffled-MFM placebo) and the result barely moves: **70%** of nonsense oscillators match or beat the real one (*p* = **0.703**). The volume/MFM geometry carries no information. |

> **In one sentence:** the Chaikin oscillator looks predictive because indices drift up — encode it mechanically (standard EMA3−EMA10 of the A/D line, cross above zero, no eyeballing) and fire the rule 1,452 times across 5 indices over 21 years, and it **loses to buying on random days** at every horizon (and is *significantly worse* than random at 60 days, *p* = 0.03; the volume placebo leaves the result untouched, *p* = 0.70): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The **Chaikin Oscillator** is
the standard `EMA3(ADL) − EMA10(ADL)`, where the Accumulation/Distribution Line cumulates volume
weighted by the Money Flow Multiplier `((C−L)−(H−C))/(H−L)`; all EMAs are causal (past-only). A
long fires on the first bar whose oscillator turns from ≤ 0 to **> 0** (read on the close of *t*),
entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day
return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). Because the shared
price-cache stores no volume, a **deterministic, look-ahead-free** range-based volume proxy feeds
the ADL (it cannot fabricate a forward lead). The Signal axis is **cross vs a drift-matched
random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a
**shuffled-MFM placebo** that destroys the accumulation geometry while keeping the volume and the
marginal. Tradability charges costs on every signal. A deterministic synthetic control with a
*planted* A/D lead proves the detector is live (edge 0 → *t* = −0.37; planted lead → *t* = +16.31),
so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the A/D line and Chaikin oscillator are, why a momentum-buy on a rising market always looks good, the cross-vs-random race, and the volume scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the causal EMA3−EMA10 oscillator, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-MFM placebo, per-ticker deltas, costs, and a synthetic planted-lead control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`chaikin_oscillator/`](chaikin_oscillator/). Oscillator is EMA3(ADL)−EMA10(ADL), causal throughout; entry is the next close (one lag). Volume is a deterministic look-ahead-free range proxy (the shared cache stores OHLC only). Basket is surviving liquid ETFs — but this is a single-instrument momentum study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
