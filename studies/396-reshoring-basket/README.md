# Study 396 — Reshoring-Basket 🏗️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the reshoring tilt out-earn beyond its beta? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The raw excess vs SPY *just* clears the bar (HAC *t* = **2.07**), but strip the **beta** and the alpha is **+4.6%/yr at HAC *t* = 1.93 — under 2**; against the proper global benchmark **ACWI** the alpha is **+2.5%/yr, HAC *t* = 1.01**. The edge is **entirely pre-2018** (alpha *t* = **2.51**) and **negative in the post-2018 narrative era** (*t* = **−0.24**), and it's carried by **one single name** (ROK, *t* = 2.16), not the theme. Positive-but-fragile ⇒ WEAK. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | What survives costs is **beta, not alpha**: a high-beta industrials + EM tilt out-returns the market for free in a rising tape. Strip beta and nothing is significant — and in the era you'd actually trade the narrative (post-2018) the residual is **negative** while the plain benchmark *out-Sharpes* the basket. No NAV-scale reshoring alpha to deploy. |
| **Structural edge?** | ![Busted](https://img.shields.io/badge/Structural_edge%3F-Busted-8b949e?style=flat-square) | A **sector-beta + single-name + backward-looking illusion**: the outperformance is mostly beta, concentrated in one stock, and **reverses sign exactly when the reshoring narrative arrives**. The thesis works where it's uninformative (the market went up) and fails where it would pay (a *distinct*, narrative-driven edge). |

> **In one sentence:** the "reshoring / nearshoring" trade looks like a real edge only until you separate the two things its track record fuses — a high-beta industrials-plus-Mexico tilt (free sector beta) from any *alpha* the manufacturing-comes-home story would have to add — and once you regress out the market the alpha is **+4.6%/yr at HAC t = 1.93** vs SPY (1.0 vs the global ACWI), is carried by a **single automation name** rather than the theme, and is **significantly positive pre-2018 yet negative once the narrative actually arrives** — real-as-sector-beta, weak-as-alpha, and a mirage as a structural trade.

## What we tested

The believers' framing is that deglobalisation — tariffs, the COVID supply-chain scare, the IRA/CHIPS subsidies — is *structurally* dragging manufacturing back to North America, so a basket of **US industrials + Mexico + factory automation** should durably out-earn a plain global index. We build a transparent equal-weight proxy (**XLI / EWW / ROK**) and test it as a persistent tilt over **27.4 years** against two benchmarks (**SPY** and the global **ACWI**), entering the honest question first: not "did it go up" (almost everything did) but "did it out-earn its benchmark by *more than its **beta** explains*?" We judge it on the **CAPM alpha** (regress out the market) with a Newey-West HAC *t*, a Sharpe race, a sign-flip placebo, costs, a per-sleeve breakdown, and a pre-announced **pre/post-2018** narrative split. A deterministic synthetic control with a *planted-alpha* knob confirms the engine is faithful — and that a high-beta basket's raw excess is **not** evidence of alpha.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the basket beat the market" isn't the same as "reshoring works," how high-beta sectors out-earn for free, and why the trade died the moment it became a story — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | raw excess vs CAPM alpha (plain + HAC *t*), the SPY-vs-ACWI benchmark race, a pre/post-2018 split, an excess-of-cash Sharpe race, a sign-flip placebo, costs, per-sleeve attribution, and a synthetic faithful-engine / planted-alpha control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`reshoring_basket/`](reshoring_basket/). The basket is an explicit **3-sleeve equal-weight proxy** (XLI/EWW/ROK), edge measured as **beta-adjusted alpha** vs SPY and ACWI. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
