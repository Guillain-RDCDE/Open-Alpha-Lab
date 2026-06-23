# Study 372 — Max-Pain-Pinning 📌

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price pin to max-pain? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The literature finds a **modest single-stock** clustering toward high-OI strikes (delta-hedging), so it isn't nothing — but it's a *strike* tendency, not a *max-pain law*. Our free tape can't test the **landing-at-expiry** claim (yfinance keeps no expiry-day history), and the live snapshot shows spot a **median ~2%** off max-pain. Literature + a *synthetic-only* positive control ⇒ **WEAK**, never REAL. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The "fade toward max-pain" bet only pays in the synthetic world where we **plant** a strong pin (+2.8% gross, 86% hit at `pin=0.40`). In the real snapshot there's no observable pull to capture, and a live single-name fade eats expiry-day spreads, borrow, and the ~2% noise. **No deployable edge.** |
| **"Pinned"?** | ![Busted](https://img.shields.io/badge/Pinned%3F-Busted-8b949e?style=flat-square) | As a *general law* it's **busted**: absent in liquid index names (SPY/QQQ sit ~0.3% off), contradicted by a snapshot scattered ~2% from max-pain, and our control proves a pin only registers when hand-built. Real-as-a-footnote, busted-as-a-law. |

> **In one sentence:** max-pain pinning is a vivid expiry-day story with a thin academic core — a *modest* single-stock clustering toward high open-interest strikes — that does **not** generalise to a tradable "price lands on the max-pain strike" law: a free option-chain tape can only snapshot today's max-pain (and finds spot a median ~2% away), so the landing claim is tested on a deterministic pinning simulator where a pin only appears (paired *t* −11 → +14) once we deliberately plant it.

## What we tested

Live option chains give only **upcoming** expiries, so we can compute today's **max-pain
strike** — the settlement price minimising total option payout — but never see how those
contracts *expire*. We therefore do two things. **(1)** A real **snapshot** of 20 liquid US
underlyings (as-of 2026-06-22): how far does spot sit from max-pain *right now*? **(2)** A
deterministic **synthetic pinning control** — many simulated expiry episodes with a tunable
drag toward max-pain — where the truth is known, so we can check the engine never invents a
pin when there is none and *does* detect one when it's planted. The decisive test compares the
expiry close's distance to max-pain against its distance to the **spot-anchor** (where price
started), with a paired *t*, a label-shuffle placebo, and a cost-charged fade trade.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what max-pain is, why "the stock pins at the strike" is mostly a story, and why a snapshot already shows price scattered ~2% away — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | max-pain computation, the snapshot gap distribution, a planted-pin control with a paired *t* + label-shuffle placebo + spot-anchor baseline, and a cost-charged fade-trade |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`max_pain_pinning/`](max_pain_pinning/). The real tape is an explicit **snapshot** (one as-of date), not an expiry-day pinning panel; the landing claim is decided on a **synthetic** control. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
