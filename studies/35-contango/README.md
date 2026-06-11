# Study 35 — Contango 🛢️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do backwardated commodities out-earn contangoed ones? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The direction is enormous — front-month USO **−76%** vs the laddered USL **+4%** on the *same* crude (roll drag +5.1%/yr WTI, +8.9%/yr gas) — but on this short, violent two-curve tape the weekly roll spread's HAC *t* is only **+1.53 / +1.75**, under the desk's *t* ≥ 2 bar (our own pre-registered line). The documented premium (Gorton–Rouwenhorst 2006; Erb–Harvey 2006; Koijen et al. 2018) rests on broad cross-sections the liquid tape doesn't offer; the synthetic control (+27.6%/yr, Sharpe +1.86, null −0.28) is a *machinery* proof — the premium is wired in by construction — not market evidence. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | On the real energy tape, **no**. Timing the front-month by the curve points the right way (WTI Sharpe **+0.35** vs −0.01 buy-and-hold) but the combined book is statistically flat (Sharpe **+0.16**, HAC *t* **0.66**) with an **−83% drawdown** — the only liquid energy curves are too few and too crash-prone to harvest. Cost isn't the killer (10 bp barely dents it); the noise and concentration are. |
| **Real-tape run?** | ![Done](https://img.shields.io/badge/Done-8b949e?style=flat-square) | **Done — no paid feed needed.** Roll yield is read directly from **front-month vs 12-month-laddered ETF pairs** (WTI USO/USL, gas UNG/UNL): their return gap *is* the term-structure roll. Real numbers, fingerprinted & as-of pinned, in [docs/results.md](docs/results.md). |

> **In one sentence:** the commodity roll drag is **economically enormous** — on the real tape the front-month USO bled **−76%** while the 12-month-laddered USL on the *same* crude was **+4%** (an 80-point contango tax; natural-gas UNG is down −99%) — yet on this short, noisy two-curve tape it stays statistically `WEAK` (weekly roll spread HAC *t* +1.5–1.8 < 2), and **harvesting** it is a `MIRAGE`: a curve-timing carry book is indistinguishable from zero (Sharpe +0.16, HAC *t* 0.66) with −83% drawdowns — so the signal's value is **defensive** (don't be the sucker long the front-month in contango), not a positive-carry alpha. *(Unmodelled and asymmetric: borrow fees on the USO/UNG shorts and the funds' ~0.6–1%/yr expense ratios — both would only shave the trade further.)*

> ✅ **Real run done — and the FRED/curve-feed problem is gone for good.** Computing roll yield needs the term structure, but you don't need a paid futures feed to see it: the **front-month** energy ETF (USO, UNG) and the **12-month-laddered** one on the same underlying (USL, UNL) differ *only* in where on the curve they sit, so `laddered − front` is the realized roll cost — the famous USO bleed, straight from yfinance. The cross-sectional bucket machinery is still proved on the synthetic control ([examples/run_synthetic_demo.py](examples/run_synthetic_demo.py)); the real energy run is [examples/verify.py](examples/verify.py) → [docs/results.md](docs/results.md).

## What we tested

The desk's idea from Kakushadze & Serur, *151 Trading Strategies* (**§9.1 roll yields**, **§9.4
value/carry in commodities**). The steelman: a long futures position earns a **roll yield** as it slides
along the term-structure curve — positive when the curve is **backwardated** (front > deferred, rolls up),
negative when **contangoed** (front < deferred, rolls down) — so a book long the most-backwardated and
short the most-contangoed commodities harvests a real carry premium (Gorton–Rouwenhorst 2006; Erb–Harvey
2006; Koijen et al. 2018). We prove the engine on a synthetic 12-commodity panel with a *baked* roll-yield
premium (and a disconnected null that earns nothing), then run it for real on the **energy tape** — reading
the term-structure roll straight off front-month vs 12-month-laddered ETF pairs (USO/USL, UNG/UNL), no paid
feed required. It is the commodity sibling of
[Study 27 (Steamroller, FX carry)](../27-steamroller/) and a cousin of
[Study 29 (Hedgers-Toll, commodity COT)](../29-hedgers-toll/).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what roll yield is, why backwardation pays and contango bleeds, and the real USO vs USL story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the real energy bleed table, the curve-timing book vs buy-and-hold, the control-vs-null bucket machinery, and the carry+momentum diversification blend |

The real energy run — every number, fingerprinted and as-of pinned — is in [docs/results.md](docs/results.md);
the **beat-7 worked complement** (does a momentum sleeve diversify the carry book? — yes, blend Sharpe
beats either leg) is in [docs/extension.md](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
