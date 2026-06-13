# Study 99 — Safety-Net 🛑

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | **No threshold out-earns buy-and-hold.** Best (15%) is **+0.03 pts/yr**, 10% *matches* it, 5% loses **4.8 pts/yr**. HAC *t* on the daily return difference runs **−2.66 (5%) to −0.39 (10%)** — never a significant positive edge. The 10/15% stops *do* beat a matched random-exit coin **100%** of the time, so the timing isn't pure noise. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | As a *return improver* it fails: the best width equals buy-and-hold, and a plausibly-chosen tight 5% stop **destroys 4.8 pts/yr** to whipsaw. Any Sharpe gain is just lower equity exposure — *beta you can buy more cheaply* by holding less stock — not skill. Taxes on dozens of switches widen the gap. |
| **Cuts drawdown?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Protection is real but **entirely depends on X**: a 10% stop cuts max drawdown **+11.3 pts** (−43.9% vs −55.2%), a 20% stop cuts **only +0.5 pts**, and a 5% stop cuts some but bleeds return. Genuine only in a middle width band. |

> **In one sentence:** a trailing stop on the S&P 500 is **drawdown insurance that works at one dial setting** (≈10–15%, ~11-point cut) and **whipsaws at a tight one** — but at no width does it **make you more money** than just holding, exactly as Kaminski & Lo (2014) predict for a long-biased, mean-reverting-ish index.

## What we tested

The most ubiquitous rule in retail trading, stated at full strength: *"Always use a stop-loss — **cut your losses and let your winners run**. A trailing stop (exit when price falls X% from its peak) protects your capital, limits drawdown, and improves returns."* We take it literally — hold SPY (total return), track the running peak, **exit to cash (earning 0%) the day the close falls X% below that peak**, wait a **21-day cooldown**, re-enter and reset the peak — act **one day after** the trigger, charge **5 bps** per switch, and **sweep X = 5/10/15/20%**. Each X is pinned against **buy-and-hold** and a **matched random-exit control** that holds the same total time in cash, in runs of the same lengths, but on random dates. Two deterministic synthetic tapes serve as the positive/negative control: a **trending** tape where stops genuinely help (escape the sustained downtrend) and a **mean-reverting** tape where they hurt (whipsaw) — the Kaminski-Lo dichotomy in a bottle.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, the X-sweep table, the equity/drawdown curves, the matched-coin test, why no width beats just holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* per threshold, the exposure-matched placebo (200 seeds × 4 X), the Kaminski-Lo trending-vs-mean-reverting controls, capacity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`safety_net/`](safety_net/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
