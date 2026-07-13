# Study 727 — Can you trade maple syrup? 🍁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does maple reward a holder, or a proxy, or a season? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The administered PPAQ bulk price grew **+2.0%/yr** (2008–2024) vs **+4.5%/yr** for the S&P/TSX — annual excess **−3.0%/yr** (*t* = −0.91). The one buyable proxy (`RSI.TO`) has alpha *t* = **1.83** (< 2, and it's defensive-sugar beta, not maple). Sugaring-season (Feb–Apr) spread **+0.58%/mo** (*t* = 0.81). |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is **no maple market** — no exchange, no futures, a committee-set price. The only buyable expression is a low-β sugar refiner; a sugaring-season timer nets **8.78%/yr**, *below* just holding it (**9.38%**). No instrument to size into. |
| **Just a curio?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | A cartel, a ~100 M-lb strategic reserve, a ~C$18.7 M heist — and no tradable edge. The loudest "seasonal" month is **July**, which has nothing to do with sugaring. |

> **In one sentence:** Quebec really does run a strategic maple-syrup reserve, but the price it defends is *administered* (near-flat, ~2%/yr, below stocks), there is no maple instrument to buy — the closest listed name is a defensive sugar stock whose mild edge is misattributed to maple — and the sugaring-season seasonal is noise (spread *t* = 0.81, bootstrap CI straddles zero); a wonderful story, not a trade.

## What we tested

The folklore treats Quebec's [Global Strategic Maple Syrup Reserve](https://ppaq.ca/en/selling-buying-maple-syrup/the-strategic-reserve/) — the barrel stockpile the producers' cartel (PPAQ) uses to defend the bulk price, famously raided in the [2011–12 heist](docs/references.md) — as proof that maple is a real, ownable soft commodity with a natural sugaring rhythm. Because there is **no maple exchange or futures**, we are transparent: we (a) hardcode a small, **clearly-cited, approximate** annual PPAQ bulk-price series (CAD/lb, 2008–2024 — a *labelled proxy*, never a feed) and test whether holding it beats the S&P/TSX; (b) test the only **tradable** name with real maple exposure, **Rogers Sugar (`RSI.TO`)**, plus a sugar-futures (`SB=F`) placebo, for Newey-West alpha vs the TSX; and (c) test a **sugaring-season (Feb–Apr)** seasonal with per-month HAC *t*-stats, a Welch test, a block-bootstrap CI and a costed timer. (Same labelled-proxy shape as [Study 358](../358-watch-index/); same seasonality machinery as [Study 307](../307-coffee-seasonality/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an *administered* price looks flat, the stocks-vs-maple race, why the "buyable" trade is really a sugar stock, and the season that isn't — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | maple vs TSX annual-excess *t*, Newey-West proxy alpha (+ the misattribution), per-month HAC *t* & a Feb–Apr Welch test with a block-bootstrap CI, a costed timer, and a synthetic-season positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`maple_syrup_reserve/`](maple_syrup_reserve/). Maple price is a **hardcoded, cited, approximate proxy** — not a live feed; equity tickers are **labelled proxies** for the trade, not the price of a barrel of syrup. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
