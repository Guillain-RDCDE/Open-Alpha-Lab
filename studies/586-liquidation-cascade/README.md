# Study 586 — Liquidation-Cascade 💥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the post-liquidation bounce real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **Synthetic-only** — there is *no free* forced-liquidation tape (Coinglass/Amberdata are paywalled), so a `REAL` stamp (robust *t* ≥ 2 on a **real** tape) is unreachable. On the synthetic **null** world the engine finds **no bounce**: 5-day forward after a top-5% liquidation day **−1.20%** vs baseline **−0.29%**, gap **−0.96%** (two-sample *t* **−0.92**, placebo *p* **0.34**), sign **flips** across the horizon × threshold grid. |
| **Tradability** — does buying the blood pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | An unreachable data feed, a sign-unstable gap that is negative on the headline cut, and a 20 bps crypto round-trip that turns even the best (15-event, *t* 1.27) synthetic cut into a curve-fit. Nothing to harvest. |

> **In one sentence:** "buy the blood after a big liquidation cascade" is a plausible fire-sale
> story (mechanical selling overshoots, then bounces) — but the forced-liquidation tape it needs is
> a paid product with no free history, so on a no-key stack it is **untestable on real data**, and
> on the deterministic synthetic **null** world the engine finds no bounce (gap −0.96%, *t* −0.92,
> placebo *p* 0.34) with a sign that flips across horizons and thresholds; the positive control
> confirms the same engine banks a *planted* bounce past *t* = 2, so this is an honest null, not a
> broken detector.

## What we tested

The **capitulation-bottom folklore**: a large wave of **forced liquidations** — over-leveraged
crypto longs margin-called and dumped by the exchange — is *mechanical* selling that overshoots to
the downside, so a liquidation spike marks a local bottom and the price **bounces** (Shleifer-Vishny
fire-sales; Coval-Stafford's forced-selling *reversal*). We build a deterministic synthetic
BTC-style tape with an explicit forced-liquidation channel and one knob (``bounce_alpha``: bounce /
null / falling-knife), then run an **event study** — forward returns after top-quantile liquidation
days vs the unconditional baseline, a two-sample *t*, a **label-shuffle placebo**, a horizon ×
threshold robustness sweep, costs, and a seed-robust synthetic positive control that plants the
bounce and proves the engine catches it. **The real forced-liquidation series (Coinglass/Amberdata)
is a paid product with no usable free history — the data-availability limitation is named on the
SIGNAL axis, and it caps this study at `WEAK`/`NONE`.** *Distinct from the desk's other crypto
studies — [133 crypto-seasonality](../133-crypto-seasonality/) / [175 crypto-weekend](../175-crypto-weekend/)
(calendar), [210 crypto-trend](../210-crypto-trend/) / [251 crypto-reversal](../251-crypto-reversal/)
(price-only), [325 crypto-fear-greed](../325-crypto-fear-greed/) (sentiment): this one keys on
**forced-liquidation flow**, a microstructure channel none of those touch.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a liquidation cascade is, why "buy the blood" sounds smart, why we can't test it for free, and what the null world shows |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the event study with a two-sample *t*, the placebo null, the horizon × threshold sign-flip, costs, and the seed-robust synthetic positive control |

The reproducible headline run (synthetic **null** world, 1,500 days, frame fp `b3d7960bf851`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery lives in
[`liquidation_cascade/`](liquidation_cascade/) and runs fully offline and deterministic.

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`liquidation_cascade/`](liquidation_cascade/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
