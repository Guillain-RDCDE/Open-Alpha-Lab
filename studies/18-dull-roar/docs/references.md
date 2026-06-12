# References & literature map — Study 18 (Dull-Roar)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); full text also on
  arXiv ([1912.04492](https://arxiv.org/abs/1912.04492), Spanish edition, identical formulas). This is
  the desk's first study drawn from a *catalogue* rather than a single paper. The relevant entry is
  **strategy §3.4, "Low-volatility anomaly"**: define σ_i as a stock's historical volatility (6–12
  month lookback), then build a dollar-neutral book **long the bottom-σ decile, short the top-σ
  decile**. The book describes; it does not backtest — so the entry is a hypothesis, which is exactly
  what we put through the protocol.

## The claim under test — the steelman

- **The low-volatility / idiosyncratic-volatility anomaly.** Andrew Ang, Robert Hodrick, Yuhang Xing &
  Xiaoyan Zhang, *"The Cross-Section of Volatility and Expected Returns"*, **Journal of Finance** 61(1),
  2006 (and the 2009 international follow-up). The empirical finding we steelman: stocks with *higher*
  past (idiosyncratic) volatility earn *lower* subsequent returns — the opposite of the risk-reward
  intuition.

- **Betting Against Beta — the mechanism.** Andrea Frazzini & Lasse Heje Pedersen, *"Betting Against
  Beta"*, **Journal of Financial Economics** 111(1), 2014. The economic story behind the flat
  security-market line: leverage- and margin-constrained investors who want higher returns bid up
  *high-beta* assets (rather than lever a low-beta portfolio), so high-beta is overpriced and low-beta
  underpriced. The corrective trade is the **beta-neutral** long-short — lever the low-beta leg up and
  the high-beta leg down — which is precisely the construction `decompose.beta_neutral_bab` implements
  and the synthetic generator bakes in (`alpha_i = -s·(β_i − β̄)`).

- **Low risk as an investment style.** Malcolm Baker, Brendan Bradley & Jeffrey Wurgler, *"Benchmarks
  as Limits to Arbitrage: Understanding the Low-Volatility Anomaly"*, **Financial Analysts Journal**
  67(1), 2011; David Blitz & Pim van Vliet, *"The Volatility Effect"*, **Journal of Portfolio
  Management** 2007. Why the anomaly persists (benchmark-relative mandates discourage the low-beta
  trade) and how a practitioner harvests it long-only.

- **The lottery-stock pattern.** Turan Bali, Nusret Cakici & Robert Whitelaw, *"Maxing Out: Stocks as
  Lotteries and the Cross-Section of Expected Returns"*, **Journal of Financial Economics** 99(2), 2011.
  High-vol names attract lottery-seeking demand and rich idiosyncratic vol — the reason our synthetic
  sets `ν_i ∝ β_i` (idio vol rises with beta) and the reason the short leg is the expensive one.

## The honest counters — why the verdict is `WEAK` / `MIRAGE` / `BETA-TILT`

- **Most of "low-vol alpha" is low beta / known factors.** Clifford Asness, Andrea Frazzini & Lasse
  Pedersen, *"Low-Risk Investing Without Industry Bets"* (2014) and the broader factor-zoo critique
  (Hou, Xue & Zhang 2015; Harvey, Liu & Zhu 2016): a long-only low-vol book is largely a low-beta,
  value- and quality-tilted position. `decompose.beta_tilt_test` prices exactly this — lever the
  long-only book to β=1 and read the residual.

- **The effect concentrates in the unshortable short leg.** The Ang et al. result is strongest among
  small, illiquid, hard-to-borrow high-vol names. Gene D'Avolio, *"The Market for Borrowing Stock"*,
  **Journal of Financial Economics** 66, 2002, documents how those are exactly the expensive- or
  impossible-to-borrow stocks — the friction `strategy.borrow_sweep` / `extension.borrow_breakeven`
  charge against, and the reason the dollar-neutral form is a `MIRAGE`.

- **Post-publication decay in US large caps.** A recurring finding (e.g. McLean & Pontiff, *"Does
  Academic Research Destroy Stock Return Predictability?"*, **Journal of Finance** 2016) is that
  published anomalies weaken after publication. Our current-S&P-500, 2010→ sample — a high-beta-friendly
  decade — shows the low-vol edge essentially absent, consistent with decay (and with our own
  survivorship caveat below).

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference.** Whitney Newey & Kenneth West, *Econometrica* 55(3), 1987 — the
  autocorrelation-robust standard errors behind every *t*-stat here (`decompose._ols_nw`).
- **Reproducibility.** The headline numbers are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (an explicit as-of date + a content fingerprint of the input panel), so a re-run that matches the
  fingerprint holds the same tape. The cross-section is built with
  [`quantlab.universe`](../../../quantlab/universe.py).

## Caveats stated in the open (house rule)

- **Survivorship bias.** The real panel uses *current* S&P 500 membership, which excludes delisted
  names — biasing the surviving high-vol leg upward (today's wild winners survived; the wild losers
  left). The sample is therefore **structurally hostile to this anomaly**: the blow-ups the short leg
  is supposed to harvest are exactly the names the panel removed, which is part of why the per-leg
  alphas *invert* here (wild leg positive). The long-run academic effect lives in delisted-inclusive,
  all-cap data (a beat-7 fork).
- **Window.** 2010→ is one regime (a long, high-beta-favouring bull), and the panel's start date —
  set by data availability, not by us — excludes 2008-09, the very crash where defensive low-vol
  earns its keep. The `WEAK` stamp records that the effect is real in theory and on our control but
  absent-to-inverted *here* — a statement about fragility, not a clean refutation.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
