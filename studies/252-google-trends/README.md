# Study 252 -- Search-Trends

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Da-Engelberg-Gao (2011) document a real attention->reversal effect at *weekly* horizon on a broad retail universe. But on this monthly, mega-cap, proxy-Trends tape the long-low/short-high (fade-the-hype) spread is **-77%/yr at HAC *t* = -1.34** -- wrong sign and not significant. Bootstrap spread-Sharpe CI = [-1.64, +0.02] (97% negative). Literature support *without* a robust real-tape HAC *t* >= 2 is **Weak**, not Real. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross spread already wrong-signed; the short leg is the meme tercile (GME/AMC/COIN/RIVN/LCID) -- exactly the hard-to-borrow, squeeze-prone names where 8 bps/mo borrow is fantasy; only 30 survivor mega-caps over 48 portfolio months; attention is a hand-curated proxy. Nothing here is implementable. |
| **Survivorship + proxy bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | Basket = fixed mega-cap survivors; attention table = stylised hand-curated approximation of Google Trends. All results are upper bounds on a real, broad-universe, true-SVI implementation. |

> **In one sentence:** the famous "Google searches predict returns" idea is real in its original weekly, broad-universe, reversal form -- but on a monthly mega-cap basket with a curated Trends proxy it produces a wrong-signed, statistically insignificant spread (*t* = -1.34) whose short leg is the un-shortable meme zoo, making it a textbook *Weak / Mirage*.

## The claim

> *Do spikes in Google searches for stock names front-run their returns?*

## What we tested

Each month we standardise every stock's curated Google-Trends-style search
interest against its own trailing 6-month baseline (an *attention z-score*: how
abnormal is this month's interest?). We then sort a basket of 30 household-name
stocks: LONG the low-attention tercile, SHORT the high-attention tercile, hold
one month, rebalance. The reported spread is **low minus high** -- positive when
the Da-Engelberg-Gao (2011) attention->reversal ("fade the hype") effect pays,
negative when riding the hype (momentum) pays. We pin the result against (a) the
equal-weight basket, (b) a random-portfolio control of identical tercile size,
(c) a sub-period split, and (d) a turnover + short-borrow cost sweep. A
deterministic synthetic positive control confirms the engine recovers a planted
attention->reversal premium (*t* = +12.6 at premium = 0.03).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the attention proxy and its iconic spikes (GME, ZM, NVDA, WMT), the fade-vs-ride bet in plain language, why the folk story breaks down, and the wrong-signed headline |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | attention z-score construction, synthetic positive control, HAC *t*-stats, bootstrap Sharpe CI, sub-period sign-flip, turnover + borrow cost sweep, random-portfolio null, proxy/survivorship caveats |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`google_trends/`](google_trends/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
