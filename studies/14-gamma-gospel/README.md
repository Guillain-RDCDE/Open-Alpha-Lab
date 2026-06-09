# Study 14 — Gamma-Gospel 🃏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the GEX sign forecast the day's character? | ⏳ `WEAK` *(pre-registered)* | Negative-gamma days really are more volatile and more trending — but those are the **high-VIX** days. The pitch lives or dies on whether the gap **survives controlling for VIX**; the desk's prior (and [Study 12](../12-paper-prophet/)) says the range-vol leg is near-tautologically VIX, leaving directional efficiency as the only place GEX might keep information of its own. The real SPY chain settles it. |
| **Tradability** — could you get paid? | ⏳ `MIRAGE` *(pre-registered)* | Even a surviving sign is a **bias on the day's character, not an entry** — to express a few points of trend-vs-chop tilt you still pay a round-trip options/index spread on a whole-day hold. The pitch itself concedes it's *"a shift, not a re-architecture."* |
| **Built on an unobservable?** | ⚪ `ASSUMED` | GEX is not measured, it's *assumed*: the SqueezeMetrics dealer convention (long calls / short puts). Flip the assumption and the whole map inverts — the single biggest modelling risk, flagged not hidden. |

> **In one sentence:** the GEX "regime read" is a **real but VIX-shadowed dealer-gamma effect dressed as a crystal ball** — negative-gamma days are wilder mostly *because they're high-VIX days*, the sign is a context bias rather than a trade, and the whole map rests on an unobservable assumption about who's holding what; the offline core proves our VIX-control test can tell a genuine effect from the relabel, and `verify.py` points it at the real SPY chain.

## What we tested

A viral thread ([GEX Edge](https://gexedge.io), *"Gamma Exposure Explained,"* June 2026) argues the *input* to price is **dealer hedging**: sum every option's gamma × open-interest (calls long, puts short — the SqueezeMetrics convention) into net **GEX**, and its *sign*, knowable before the open, tells you the day's character — **positive** ⇒ dealers fade moves ⇒ a calm **range** day; **negative** ⇒ dealers chase ⇒ a violent **trend** day, *"more important than direction."* We steelman it (dealer gamma genuinely relates to realised vol), then build a daily panel — GEX at the prior close from the real **SPY** option chain (Alpha Vantage `HISTORICAL_OPTIONS` — one of the few sources carrying **both open interest and gamma** historically, though its endpoint is **premium**; the free options sources we checked carry no usable historical OI), the next session's **range vol** and **directional efficiency** from daily OHLC, and the prior-close **VIX** — and ask the one question that decides it: *does the GEX sign add anything **over the VIX**, or is it the volatility regime in a trenchcoat?*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the dealer-hedging story, the VIX hiding under the regime, and why the *raw* gap is a trap |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the GEX construction + its dealer assumption, the HAC nested regression, the baked-in-β recovery, and the trenchcoat collapse |

The pre-registered test and the offline validation are in [docs/results.md](docs/results.md); reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py), then earn the real-tape stamps with [examples/verify.py](examples/verify.py) (`--fetch` with a paid options-chain key — Alpha Vantage premium or equivalent; the stamps stay ⏳ pre-registered until that real run lands).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
