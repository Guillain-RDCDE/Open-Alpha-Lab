# Study 744 — Tetraphobia 🔢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The "unlucky" **4/4** date does not underperform: pooled EWT+EWH+MCHI **+16.2 bps**, *t* = **+0.69** (wrong sign — every market is *positive*), random-calendar placebo left-tail *p* = **0.748**. No return footprint at all. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Shorting the 4/4 session (betting on the drop) loses **−27.2 bps/event** net (*t* = −1.16): the day tends to *rise*, and costs finish it. |
| **Do Asian prices dodge the 4?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | On live tape, Greater-China prices avoid a trailing **4** and prefer **8**: Asia *z*(8 > 4) = **+4.73** (Taiwan +5.25, China A-shares +3.62); the US control is flat (−0.84). The superstition is measurably printed into prices. |

> **In one sentence:** tetraphobia is real in the *ticks* and absent in the *returns* — Greater-China prices genuinely dodge the unlucky 4 and reach for the lucky 8 (*z* = +4.73, US control flat), but the "unlucky 4/4" date doesn't fall at all, and shorting it just bleeds costs.

## What we tested

The East-Asian fear of the number 4 — a homophone of *death* — is documented everywhere
from skipped building floors to discounted licence plates, and there is a real academic
finding ([Brown & Mitchell 2008](docs/references.md)) that Chinese stock **prices** avoid
a trailing 4 and prefer 8. The folklore adds a tradable claim: the maximally-unlucky
**4/4** calendar date should see those markets sell off. We test both halves — a
trailing-digit clustering test on raw local-currency closes for a Taiwan/HK/China/Korea
basket against a US-control placebo, and a one-sample-*t* event study of the 4/4 session
return across 2000→2025 in EWT/EWH/MCHI with a random-calendar placebo, an 8/8
"lucky-date" contrast, and a costed short.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the superstition, the digit histogram that really does dodge 4, and why the "unlucky day" trade is a mirage |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-proportion z-battery on trailing digits (by region, vs a US placebo), the 4/4 one-sample-*t* + random-calendar placebo, the 8/8 contrast, the costed short, two synthetic controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tetraphobia/`](tetraphobia/). Clustering is **price-only** (raw traded closes,
by design — an adjustment destroys the last digit); calendar returns are **total-return**.
The **US basket is the placebo**; Korea contributes no trailing 4/8 digit (won-priced) —
named, not hidden. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
