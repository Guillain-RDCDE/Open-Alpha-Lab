# Study 311 — Government-Shutdown 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Naive HAC *t* clears 2 at long horizons (+2.33 at 20 sessions), but only against zero — the excess over a random date isn't significant (permutation *p* = 0.13) and rests on **5 events**. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "edge" is just the market's normal drift (+0.9% on a random month too); costs are irrelevant because there's no abnormal return to bank. |
| **Buy the dip *every* time?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | One of the five was negative (−4.0% in 2018), the headline is carried by a single Christmas-Eve 2018 coincidence, and the trade dies if you enter one day late. |

> **In one sentence:** stocks usually rise after a government shutdown — but no more than they rise after *any* random date, and five events is far too few to call it a strategy.

## What we tested

The recurring market take, every time Washington runs out of funding: *"the S&P 500 rose during most past shutdowns and bounced back fast — shutdowns are a buying opportunity"* (CNBC, LPL Financial, Reuters wrap-ups). We take it at full strength as a clean **event study**: line up every modern US federal funding-gap shutdown (a hardcoded, CRS-sourced table), buy SPY total-return at the close of the day each one starts, hold a fixed horizon, and measure the forward return — then race it against a **synthetic event-null** (the same horizon around 20,000 random dates) and a +1-session look-ahead check. SPY total-return daily tape back to 1993; the offline core and tests run on a deterministic synthetic generator that plants (or withholds) a post-event bounce.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language, the random-date comparison, the one-day-late tell |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the CAR curve, naive *t* vs permutation *p*, lag robustness, the synthetic positive control |

The fingerprinted real run lives in [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
