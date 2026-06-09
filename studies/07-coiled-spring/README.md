# Study 07 — Coiled-Spring 🌀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout beat just holding the same name? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Buying the breakout and holding 10 days beats a random same-stock entry by **+1.2%** (20 days: **+1.5%**), but the HAC *t* is only **2.0–2.3** and **0** at 5 days — a *whisper* of real short-term momentum, not the book's fireworks, and not corrected for the universe of TA rules one could have searched. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The tradable rule nets **+0.58%/trade** after 15 bps — but the **median trade *loses* (−0.25%)** and the win rate is **41%**; the positive average is a thin right tail (per-trade Sharpe **0.05**, CI [0.002, 0.090]). It's really **long-momentum-regime beta** and **dies by ~75 bps round-trip** — a cost level the *small-caps the book actually trades* would blow straight through. |
| **Explosive as advertised?** — the +30-50%-in-days promise | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Across all **1,674** breakouts, the share that gross **≥ +30%** is **1.7%** — a **1-in-60 lottery**, not the base rate. Median hold is **3 sessions**, median outcome a small loss. The book's pitch is the cherry-picked tail. |

> **In one sentence:** the 20-EMA pivot breakout carries a faint, real pulse of short-term momentum — about **+1% over ten days** beyond simply holding the stock — but the book's "explosive +30-50% in days" is a **1-in-60 tail event sold on survivors**: the median trade loses, the positive average is bull-market beta, and on the small names it targets, ordinary costs finish it.

## What we tested

A retail trading-book rule, stated at full strength: *"A stock that pulls back to its rising 20-EMA and then breaks its pivot high on big volume is about to explode — +30 to +50% in 6 to 10 days."* The claim comes from Jayesh Shah's self-published *Trade the 20 EMA* (see [docs/references.md](docs/references.md)) — three hard steps with no fitted parameter: form a 20-EMA pivot, hold the EMA on the pullback, then buy the pivot breakout on at least 2× the prior month's average volume. We mechanise every word and run it over a cached, liquid 174-name US universe back to 1962, counting every breakout — winners *and* losers.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the teardown |

Reproduce the headline run via [`examples/verify_real.py`](examples/verify_real.py) → [docs/results.md](docs/results.md) (as-of + fingerprint).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
