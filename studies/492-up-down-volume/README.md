# Study 492 — Up-Down-Volume 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the up/down-volume ratio forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the selling climax" rule does **not** beat a drift-matched **random-entry** baseline at the desk's bar: climax − random = **+27.9 / +28.1 / +36.0 / +77.4 bps** at 5/10/20/60 days — positive at every horizon, but the climax-vs-random Welch *t* **never clears 2** (max **+1.95** at 5d, *p* = 0.052, just misses). The big one-sample *t*'s (20d **+3.67**, 60d **+4.50**) are **beta + the post-drop bounce** every dip-buy inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The small positive tilt isn't separable from "buy after a drop and hold" — which needs no breadth feed — and costs erode it. You'd capture the same drift + mean reversion more cheaply by **holding the index** (or buying any dip). Nothing reliable to scale. |
| **"Does volume breadth forecast?"** | ![Busted](https://img.shields.io/badge/Volume_breadth_forecasts%3F-Busted-8b949e?style=flat-square) | The timing placebo *is* significant (**p = 0.028** — climaxes really cluster in drawdowns), yet that does **not** survive the drift-matched random-day test (*t* < 2 at every horizon). The up/down ratio rediscovers the index's own short-horizon mean reversion; it adds no tradable foresight. |

> **In one sentence:** "Up vs down volume" looks prophetic because a selling climax fires after a drop and indices both drift up *and* bounce after drops — build it mechanically from a 10-ETF breadth basket, fire the climax-buy 643 times over 21 years, and it **fails to beat buying on random days** (Welch *t* ≤ 1.95 everywhere); the dates do cluster in drawdowns (placebo *p* = 0.028) but that's the index's own mean reversion, not breadth foresight.

## What we tested

We encode the tightest mechanical version a proponent would accept. From a basket of liquid SPDR sector ETFs (SPY + XLK XLF XLE XLV XLI XLY XLP XLU XLB — a **proxy** for exchange-wide advance/decline volume) we form the daily **up-volume share** `uvs = up_vol/(up_vol+down_vol)`; a **selling climax** is a day whose share is at/below its rolling-60-day **10% quantile** (computed on *past* bars only, no look-ahead); a long fires on the first climax of each run, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY (yfinance daily total-return, 2005→2026). The Signal axis is **climax vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-volume timing placebo** that permutes the breadth series in time (marginal kept). Tradability charges costs on every climax. A deterministic synthetic control with a *planted* selling-climax bounce proves the detector is live (edge 0 → *t* = +0.46; planted bounce → *t* = +7.89), so the flat real-tape result is a genuine "nothing tradable".

> ⚠️ **Breadth proxy caveat.** A 10-ETF basket cannot reproduce the advance/decline volume of thousands of listed issues (the true Arms/TRIN input). This caps the test — it's the honest, fully-cached, reproducible version, and a floor rather than a ceiling on the method.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what up/down volume is, why a selling-climax buy on a rising market always looks good, the climax-vs-random race, and the timing scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical up-volume share, one-sample HAC *t* vs the beta+bounce trap, the random-entry Welch test, the shuffled-volume timing placebo, per-instrument deltas, costs, and a synthetic planted-climax control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`up_down_volume/`](up_down_volume/). Breadth is the up-volume share across a 10-ETF basket (proxy for exchange up/down volume); the climax threshold is a past-only rolling quantile; entry is the next close (one lag). Forward returns on SPY; the random-entry baseline neutralizes the drift. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
