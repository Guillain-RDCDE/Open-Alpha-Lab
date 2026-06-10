# References & literature map — Study 28 (Carousel)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §4.1 (sector momentum
  rotation)** — rank a set of sector ETFs by trailing momentum and rotate into the leaders.

## The claim under test — the steelman

- **Industry / sector momentum.** Tobias Moskowitz & Mark Grinblatt, *"Do Industries Explain
  Momentum?"*, **Journal of Finance** 54(4), 1999: industry portfolios exhibit significant momentum, and
  an industry-momentum strategy is profitable — much of individual-stock momentum, they argue, is an
  industry effect. This is the academic backbone for "the hot sector stays hot".

## The honest counters — why the verdict is `WEAK` / `MIRAGE` / `Not supported`

- **The right benchmark is the diversified basket.** A long-only rotation book of a few sectors has a
  beta near 1 to the equal-weight sector basket, so most of its return is just the market. The decisive
  question is the *alpha over the basket* — and on 11 liquid SPDRs it is indistinguishable from zero.
- **Sector momentum is weak in a small, liquid cross-section.** The Moskowitz–Grinblatt result uses 20
  industries over a long pre-2000 sample; on 11 heavily-arbitraged SPDR ETFs post-1998 the long-short
  factor is flat-to-negative — consistent with the broad post-publication decay of momentum strategies
  (McLean & Pontiff, *Journal of Finance* 2016).
- **Parameter / concentration data-mining.** A rotation that beats the basket only at one hand-picked
  number of held sectors is a fitted artefact; `extension.topk_sweep` shows the win-rate across `top_k`
  is a coin flip. The general hazard is technical-rule data-snooping (Sullivan, Timmermann & White,
  *Journal of Finance* 1999).

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference** (Newey & West, *Econometrica* 1987) on the rotation alpha and the
  long-short factor; reproducibility via [`quantlab.repro`](../../../quantlab/repro.py).

## Caveats stated in the open (house rule)

- **A tiny cross-section.** 11 SPDR sectors is a small universe (XLRE and XLC list only in 2015/2018), so
  the top-3/bottom-3 split is coarse — sector momentum may need the breadth of 49 Fama-French industries
  to exist; that's a stated beat-7 fork.
- **Split-only closes, equal-weight basket benchmark.** Stated, not hidden; a cap-weight or risk-parity
  basket is an alternative bar.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
