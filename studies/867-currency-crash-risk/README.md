# Study 867 — Currency Crash Risk 💥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — are high-carry currencies more negatively skewed (Brunnermeier-Nagel-Pedersen crash risk)? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The crash-skew signature is **genuinely present and correctly signed**: the long-high / short-low carry basket is deeply negatively skewed (**−1.39**), tied to the carry ordering (label-shuffle *p* = **0.034**); the skew-carry cross-section is strongly monotone (Spearman **−0.83**, permutation *p* = **0.008**) and stable in sign across both eras; the crash accounting is textbook (worst week **−13%**, max DD **−36%**, worst-5% weeks **−174%/yr** vs calm **+13%/yr**). **But** the strict green bar — a robust Newey-West \|t\| ≥ 2 holding across sub-eras — is **not cleared**: the conservative NW *t* on the basket's own skewness is only **−1.51** (−1.21 / −1.92 by era; the skew *t* is structurally low-powered against rare crashes), and the cross-sectional slope *t* falls to −1.58 late. Real in sign and shape, borderline in significance. *Survivorship: fixed current membership (no de-pegged legs) — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for the crash risk? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The premium the crash is supposed to compensate is **weak to begin with** (+3.29%/yr gross, NW *t* = **+1.73**) and **collapses to +0.71%/yr** (Sharpe 0.07, *t* = 0.37) at a modest 50 bps/yr borrow — you are paid almost nothing to carry a **−36%** drawdown and a **−13%** worst week. Picking up pennies in front of the steamroller. |

> **In one sentence:** the Brunnermeier-Nagel-Pedersen carry-crash tale is **real in shape** —
> high-carry currencies really are more negatively skewed and the carry basket "goes up by
> the stairs, down by the elevator" (skew −1.39, Spearman −0.83) — but the formal Newey-West
> significance is **borderline** (|t| ≈ 1.5) and the premium you are paid for the tail is a
> mirage after costs, so the honest read is **crash real, signal weak, paycheck a mirage**.

## What we tested

Brunnermeier, Nagel & Pedersen (2008), **"Carry Trades and Currency Crashes"**: high-carry
currencies are exposed to **crash risk** — their returns are **negatively skewed** ("up by
the stairs, down by the elevator"), the higher the carry the deeper the skew, and a
long-high / short-low carry basket inherits that crash tail. We test both halves on a
**weekly 8-currency-vs-USD tape (yfinance daily FX resampled weekly, 2003-12-12 →
2026-06-26)** including the notorious high-carry **MXN**: the **skew-carry cross-section**
(does higher carry predict more negative realized skew — OLS slope + Spearman with a
permutation *p*) and the **basket crash skew** (Newey-West *t* on the standardised-cubed
residuals, a label-shuffle placebo, a crash-conditional split, a two-era cut, a costed
timer, and a 20-seed synthetic positive control). Carry is a transparent per-currency
**proxy** (rate differentials are not on yfinance). The basket is a **current** fixed
membership survivor set — named on the **Signal** axis. **Dedup:**
[364-fx-carry-trade](../364-fx-carry-trade/) tests the carry **premium** (does UIP fail),
not the crash skew that justifies it; [828-fx-dollar-factor](../828-fx-dollar-factor/)
tests the common **dollar factor** DOL, not the high-minus-low carry cross-section;
[27-steamroller](../27-steamroller/) is the generic sold-insurance archetype, this the
specific FX instance; [797-fx-value-ppp](../797-fx-value-ppp/) tests the **PPP value**
currency signal, not carry. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the carry trade goes up by the stairs and down by the elevator — and the live synthetic crash control |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the basket-skew Newey-West *t*, the skew-carry slope + Spearman, the label-shuffle placebo, the crash split, the two-era cut, the cost sweep, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`fx_crash/`](fx_crash/). Weekly FX pulled via yfinance into this study's own
`_cache/` (fixed current membership → magnitudes are an upper bound). Carry is a documented
per-currency proxy. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
