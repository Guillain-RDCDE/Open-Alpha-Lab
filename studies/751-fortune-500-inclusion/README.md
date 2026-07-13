# Study 751 — Fortune-500-Inclusion 🏆

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does making (or falling off) the Fortune 500 move the stock? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Canonical CAR[0,+2] is **+0.06%** for debuts (*t* = **0.04**) and **+0.12%** for exits (*t* = **0.06**) — a **−0.06pp** gap at Welch *t* = **−0.02**, with an exactly-coin-flip **50%** win-rate in *both* buckets. A random draw beats the "added" pop **97%** of the time (placebo *p* = **0.97**), and even the un-tradable reveal *day* is a flat **−0.47%**. No window clears *t* = 2. **Survivorship** named on this axis (the worst exits went bankrupt and left the tape). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to trade. The most favourable of several windows (enter +1 day, hold five) reaches only **+2.23%** at *t* = **1.39** before it is window-mined, the added−dropped spread is *t* = **0.20**, and costs (not even binding) push it the wrong way. |
| **"Attention effect?"** | ![Not supported](https://img.shields.io/badge/Attention_effect%3F-Not_supported-8b949e?style=flat-square) | The *real* index-inclusion effect (S&P 500) runs on a **forced-buying demand shock** a magazine ranking simply doesn't create — no fund tracks the Fortune 500 and the deciding revenue was already public. Absent the mechanism, prestige alone moves nothing: zero, in every window, on both legs. |

> **In one sentence:** the S&P-500 index-inclusion pop is real because index funds are *forced* to buy — the Fortune 500 is a magazine ranking by already-public revenue that forces no one to do anything, so across a transparent table of 26 debuts and exits the reveal-window abnormal return is **+0.06%** for adds and **+0.12%** for drops (a −0.06pp gap at *t* = −0.02, placebo *p* = 0.97, 50% win-rate both sides, even the reveal day flat), and a synthetic control confirms ~a dozen events per bucket couldn't detect a plausible edge anyway — a picture-perfect null wearing a prestigious cover.

## What we tested

Fortune sells no free point-in-time membership feed, so we hardcode a **transparent, cited,
labelled-proxy table** of ~26 notable Fortune-500 **debuts** (Tesla, Netflix, Uber, Airbnb,
Coinbase, Moderna, DoorDash, CrowdStrike, Robinhood…) and **exits** (Mattel, GameStop, Xerox,
Bed Bath & Beyond, Gap, Hasbro…), each snapped to that year's June list-reveal date, and run a
textbook short-window **event study**: the **cumulative abnormal return** (CAR) around each
reveal, where "abnormal" is the stock's return minus a **market-model** fit (`stock = α + β·SPY`)
on a clean pre-event window. We steelman it as the [index-inclusion trade](../249-index-inclusion/)
(Shleifer 1986; Harris & Gurel 1986) — then note the crux: that effect needs **forced index-fund
buying**, which a media ranking doesn't cause, leaving only a pure **attention/prestige** channel.
We add a placebo null sized to the event count, a one-day execution lag, cost accounting, and a
deterministic synthetic power control. The debut/exit year is a curated proxy and the worst exits
delisted — both named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the S&P-500 pop is real but the Fortune-500 one isn't, what an abnormal return is, and why a prestigious list with no fund behind it moves nothing — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model CAR by bucket, added−dropped Welch *t* + a placebo non-event-window null, the [0,0]-vs-[0,+2] window split, a 1-day-lag tradable variant + costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`fortune_500_inclusion/`](fortune_500_inclusion/). Events are an explicit **hardcoded, cited, labelled-proxy table** of Fortune-500 debuts/exits (the year is curated; the worst exits delisted — survivorship named). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
