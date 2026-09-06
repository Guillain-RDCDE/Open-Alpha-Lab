# Study 989 — The One-Way Beta 🎢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do altcoins have a higher beta to Bitcoin in falling markets than in rising ones? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Across 6 majors against Bitcoin over 8.6 years, the median altcoin's beta is **0.77 on Bitcoin's up days and 1.13 on its down days** — a difference of **+0.37**. The naive two-sample *t* on that difference averages +4.23; the block-bootstrapped one, which also accounts for the randomness in *which days count as down days*, is **+5.06** — smaller by a factor of 0.8. The corroborating measurement: 100% of the panel has negative coskewness, the third-moment fingerprint of the same phenomenon. And the control that matters most — under a symmetric simulated world with one beta, a naive split still declares asymmetry in **5%** of runs. |
| **Tradability** — does anything survive the asymmetry once it is priced? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Compounded rather than averaged, the median altcoin captured **107% of Bitcoin's up days and 109% of its down days**, and ran a maximum drawdown of -95% against Bitcoin's -83% (1.14×). That gap is large enough to matter to position sizing: the leverage you are buying is not the leverage you are paying for. |

> **In one sentence:** The median altcoin's down-beta exceeds its up-beta by +0.37, which sounds decisive until you notice that a symmetric simulated world produces the same verdict 5% of the time.

## What we tested

The pitch for an altcoin is leverage on Bitcoin: same view, more of it. Nobody
states it as a *symmetric* claim, but everybody hears it as one. So: is the beta the same on the
way down as on the way up?

Splitting a sample on the sign of the market return and fitting a beta in each half is one of
the easiest false positives in finance, and this study is built around not committing it. Three
separate things can manufacture the appearance of asymmetry out of nothing: **conditioning on
the regressor** truncates its distribution, so the two conditional betas of a perfectly
symmetric bivariate normal are not equal in a finite sample; **tail correlation rises under
normality** (Longin & Solnik 2001; Ang & Chen 2002), so the "correlations go to one in a crash"
observation is partly a conditioning artefact; and **betas drift**, so a coin whose beta rose in
a bad year shows a fake asymmetry.

Each gets its own control. The up/down difference is tested with a **block bootstrap that
re-derives the split inside every resample**, so the randomness in *which days count as down
days* is priced in — and it comes out much smaller than the naive two-sample *t* everyone
quotes. Tail correlations are reported beside the value a simulated bivariate normal produces.
Betas are refitted within eras. The phenomenon is then checked against two independent
measurements — the **Bawa-Lindenberg** and **Hogan-Warren** downside betas — and against
**coskewness**, its third-moment fingerprint. Finally, the whole procedure is run on a
symmetric simulated world to measure how often it cries wolf.
**Dedup:** distinct from **238-betting-against-beta** and **419-downside-beta-equities** (the
equity cross-section), **142-bitcoin-correlation** and **604-crypto-equity-beta** (crypto against
*equities*, not within crypto), **988-bitcoin-volatility-decay** (the level of volatility, not
its asymmetry) and **987-silver-high-beta-gold** (a beta-stability question in metals, where the
asymmetry split is a secondary cut rather than the subject).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | whether altcoins really fall harder than they rise, and the statistical trap that makes every asset look that way |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | conditional betas with a block-bootstrapped difference, Bawa-Lindenberg and Hogan-Warren downside betas, coskewness, tail correlation against a normal benchmark, era controls, and a false-positive rate measured on a symmetric world |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`onewaybeta/`](onewaybeta/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
