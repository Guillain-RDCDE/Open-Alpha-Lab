# A point-by-point response to Knuteson

This document answers the specific claims in Bruce Knuteson's overnight-returns
work (*Celebrating Three Decades of Worldwide Stock Market Manipulation*, 2019;
*Strikingly Suspicious Overnight and Intraday Returns*, 2020; *They Still
Haven't Told You*, 2022; *Nothing to See Here*, 2023). Every counter-measure
below is **computed in this repository** on the same kind of public data, and is
reproducible — see [`notebooks/02_for_the_quants.ipynb`](notebooks/02_for_the_quants.ipynb)
and [`overnight/`](overnight/).

**Our stance in one line:** the empirical fact is real and Knuteson deserves
credit for surfacing it; but the *magnitude* is oversold, what survives is
mostly risk premium that is untradable and decaying, and the *manipulation*
attribution is not supported by the evidence offered.

Legend: 🟢 we agree · 🟡 we partly agree / it's overstated · 🔴 we disagree.

---

### Claim 1 — "Across world markets, overnight returns are large and positive while intraday returns are flat to negative, for decades." 🟢

**We agree, and we don't hedge it.** On SPY, the overnight mean is
+3.25 bps/day with a **Newey-West (HAC) t ≈ 4.9**; the intraday mean is
insignificant (t ≈ 0.9). The Lo (2002) Sharpe t-stat is ≈ 4.5. The fact is
statistically robust, not a fluke of i.i.d. assumptions.
→ *§2 of the quant notebook;* [`overnight/analytics.py`](overnight/analytics.py)
`mean_tstat_hac`, `sharpe_with_se`.

---

### Claim 2 — "The cumulative effect is astronomical — hundreds, thousands, even billions of percent." 🟡

**Real arithmetic, misleading framing.** Three multipliers inflate the headline:

1. **Compounding on a log axis.** A *constant, innocent* 1 bps/night drift
   compounds to +124% over 32 years; 30 bps/night to *trillions* of percent. The
   explosion is the exponent, not a mechanism. → `diagnostics.compounding_table`.
2. **The clock illusion (the big one).** The overnight window averages **~28
   calendar hours** (it includes evenings, mornings, weekends and holidays) vs
   the **6.5-hour** trading day. Put both legs on a per-hour footing and the
   night's advantage collapses **from ~4× per session to ~1.3× per calendar
   hour**. Most of the "anomaly" is a unit error. → `analytics.time_normalized_summary`.
3. **Data artefacts.** A handful of mis-adjusted split/dividend prints
   mechanically shovels return from the day into the night; our detector flags
   them. This drives the wildest emerging-market figures. → `diagnostics.inject_split_artifact`, `flag_suspicious_returns`.

---

### Claim 3 — "This pattern is suspicious — the signature of deliberate market manipulation by a large quant firm." 🔴

**Not supported.** Framed as a likelihood ratio for the observed pattern *D*:

Λ = P(D | manipulation) / P(D | risk premium + microstructure).

- The basic night > day pattern is **highly probable under both** hypotheses
  (an overnight risk premium, investor-clientele effects, news released after
  the close, and the clock illusion all predict it). When *D* is ~equally likely
  under each, **Λ ≈ 1 — the headline pattern discriminates almost nothing.**
- The *discriminating* evidence runs the other way (Claims 4–6).

A pattern that is equally consistent with an innocent explanation is not
evidence of guilt.

---

### Claim 4 — "A firm expands its book when the market is illiquid (near the open, moving prices up) and contracts when liquid." 🔴

**The mechanism is self-defeating at the scale required to matter.** Using the
square-root market-impact law (Almgren et al. 2005), the *net* overnight edge
on SPY by capital deployed:

| Capital | Round-trip impact | Net edge |
|---|---|---|
| \$1M | ~1 bps | **+2.2 bps** |
| \$10M | ~3 bps | **≈ 0 bps** |
| \$100M | ~10 bps | **−7 bps** |
| \$1B | ~33 bps | **−30 bps** |
| \$10B | ~105 bps | **−102 bps** |

The edge is gone by ~\$10M and catastrophic at fund scale. A firm large enough
to *move world markets* would pay impact an order of magnitude larger than the
3-bps premium it is supposedly harvesting. → *§7;* `analytics.capacity_curve`.

---

### Claim 5 — "The pattern holds around the world, pointing to one global actor." 🔴

**The cross-section betrays the story.**

- **Foreign-listed ETFs invert.** UK/Germany/France/Japan ETFs (EWU/EWG/EWQ/EWJ)
  show *negative* overnight and *positive* intraday — because they trade in New
  York while their underlying markets trade during the US *night*. The split is
  relative to the **listing clock**, not a universal anomaly. One global
  manipulator cannot explain why the sign depends on where an ETF is *listed*.
- **China inverts too,** cleanly explained by the **T+1** settlement rule (Qiao
  and Dam 2020) — and our FXI proxy's overnight Sharpe (~0.26) is **not
  statistically distinguishable from zero**. → *§4.*

---

### Claim 6 — "Look at the most extreme cases (e.g. the 25 most problematic markets)." 🟡

**Selection guarantees significance.** Cherry-picking the most extreme members
of a large universe is textbook multiple testing. Across our 10 indices, under
the null you expect ~0.5 false positives at the 5% level by chance alone;
reporting only "the 25 most problematic" out of hundreds is not evidence, it is
selection. → *§4.2.*

---

### Claim 7 — Implicit: "This is a big deal / a free lunch worth acting on." 🔴

**Even granting the effect, it is neither tradable nor durable.**

- **Costs flip it negative.** Buying every close and selling every open is ~252
  round-trips/year; at a realistic 5 bps round-trip the strategy's Sharpe goes
  **negative** — exactly what liquidated the NSPY / NIWM night-effect ETFs
  (2022–2023). → `backtest.cost_sweep`.
- **It's mostly beta.** ~40% of SPY's overnight return is gap-risk beta; the
  residual alpha (~1.9 bps) sits **below** the break-even cost (~3.3 bps).
  → `stats.beta_decomposition`.
- **It's decaying.** The trailing 5-year overnight Sharpe fell from **~2 (1998)
  to ~0.5 (2026)** — textbook post-publication decay (McLean and Pontiff 2016).
  An effect fading as it becomes known is the opposite of an *ongoing* scheme.
  → `analytics.rolling_sharpe`.

---

### Claim 8 — The D.E. Shaw insinuation (citing the 2023 SEC action). 🔴

**Conflation.** The SEC's 2023 action against D.E. Shaw concerned the language
of its **whistleblower agreements (Rule 21F-17)** — not trading, and not market
manipulation. Citing it next to a manipulation thesis is innuendo, not evidence.

---

## Bottom line

| | Knuteson | This repository |
|---|---|---|
| Overnight > intraday is real | ✅ | ✅ (HAC t ≈ 5) |
| Magnitude is enormous | ✅ | 🟡 inflated by compounding + the clock + artefacts |
| It implies manipulation | ✅ | 🔴 Λ ≈ 1; discriminating evidence favours microstructure |
| One global actor | ✅ | 🔴 foreign-ETF & China inversions refute it |
| Actionable / important | implied | 🔴 untradable, mostly beta, decaying, capacity ~\$10M |

A genuine, fascinating, and *largely explained* feature of market microstructure
— wrongly dressed up as both a free lunch and a global fraud. The honest,
fully-reproducible answer (this repo) is more interesting than either.

*References, with a map of which explanation each paper argues, are in
[`docs/references.md`](docs/references.md).*
