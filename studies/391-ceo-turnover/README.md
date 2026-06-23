# Study 391 — CEO-Turnover 👔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does firing the CEO move the stock predictably? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | There *is* a real **announcement-day** bump for forced ousters (**+2.05%**, placebo *p* = **0.007**) — but it lives only in the un-holdable [0,0] window. Over any tradable window the forced bucket is **+1.02%** at *t* = **0.63**, planned is a shrug, and **forced − planned = +1.47pp** at Welch *t* = **0.76** (placebo *p* = 0.25) — **fails t ≥ 2** on ~a dozen events per bucket with a subjective forced/planned label. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The whole move is the **instantaneous repricing you learn at the close**. Enter one day later (all you can do) and the forced CAR is **−0.81%** (*t* = −0.70), worse after 10 bps. No post-announcement drift to capture. |
| **"Good news or bad news?"** | ![Busted](https://img.shields.io/badge/Good_or_bad%3F-Busted-8b949e?style=flat-square) | The folklore can't decide because the *decidable* part is unobtainable: the only reliable move is the announcement instant you can't trade, and the holdable drift is statistically **zero in both directions**. A window-mining + small-sample illusion. |

> **In one sentence:** a forced CEO ouster really does jolt the stock on the announcement *day* (+2.05%, placebo p = 0.007), but that move is the instantaneous repricing you can't trade — by the next close it's gone, the holdable forced−planned gap is +1.5pp at t = 0.76, and a synthetic control confirms ~a dozen events per bucket can't detect any CEO-turnover edge of plausible size, so it's real-as-a-day-one-reaction, weak-as-a-signal, and an un-tradable mirage.

## What we tested

True CEO-turnover databases (ExecuComp/BoardEx) aren't free, so we hardcode a **transparent, labelled table** of ~25 notable large-cap CEO changes — each tagged **forced** (ousted / pushed out) or **planned** (orderly succession) with its announcement date — and run a textbook short-window **event study**: the **cumulative abnormal return** (CAR) around each announcement, where "abnormal" means the stock's return minus a **market-model** fit (`stock = α + β·SPY`) estimated on a clean pre-event window. We compare forced vs planned CARs over the canonical [0,+2] window, add a placebo null sized to the event count, a one-day execution lag for the tradable variant, and a deterministic synthetic control with a *plantable* edge. The forced/planned label is subjective at the margin and two delisted names drop out — both named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "fire-the-CEO" has no clean answer, what an abnormal return is, where the only real move hides (the day-one jump), and why you can't trade it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model CAR by bucket, forced−planned Welch *t* + a placebo non-event-window null, the [0,0]-vs-[0,+2] window split, a 1-day-lag tradable variant + costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ceo_turnover/`](ceo_turnover/). Events are an explicit **hardcoded, labelled table** (forced/planned is the believers' framing). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
