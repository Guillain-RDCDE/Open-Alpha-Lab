# References & literature map — Study 24 (Stampede)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §3.1 (price
  momentum)** — rank stocks by past return and go long winners, short losers.

## The claim under test — the steelman

- **Cross-sectional momentum.** Narasimhan Jegadeesh & Sheridan Titman, *"Returns to Buying Winners and
  Selling Losers: Implications for Stock Market Efficiency"*, **Journal of Finance** 48(1), 1993 — the
  founding result: a 3–12 month formation, long-winners/short-losers strategy earned a large, persistent
  premium in US equities. Replicated across markets and asset classes and back to the 19th century; one
  of the most robust anomalies in the factor literature (the "12-1" convention skips the most recent
  month to avoid short-term reversal).

## The honest counters — why the verdict is `WEAK` / `FRAGILE` / `Severe`

- **Momentum crashes.** Kent Daniel & Tobias Moskowitz, *"Momentum Crashes"*, **Journal of Financial
  Economics** 122(2), 2016: because WML is short the losers, it suffers infrequent but enormous crashes
  when beaten-down stocks rebound violently (notably 2009). The strategy's left tail is fat and negative —
  `decompose.crash_profile` measures exactly this.

- **The crashes are forecastable — risk-managed momentum.** Pedro Barroso & Pedro Santa-Clara, *"Momentum
  Has Its Moments"*, **Journal of Financial Economics** 116(1), 2015: scaling the factor by the inverse of
  its own recent volatility roughly halves the crash risk and lifts the Sharpe — the overlay
  `extension.vol_managed_wml` implements, and the same machinery as [Study 16](../../16-storm-shy/).

- **Decay in modern US large caps.** The premium has weakened in large-cap US equities since publication
  (consistent with the broader post-publication-decay finding of McLean & Pontiff, *Journal of Finance*
  2016). Our current-S&P-500, 2010→ sample — survivorship-biased and growth-dominated — shows a faint,
  insignificant WML alpha, which is about *this* sample, not the long-run effect.

- **Residual momentum.** David Blitz, Joop Huij & Martin Martens, *"Residual Momentum"*, **Journal of
  Empirical Finance** 2011: momentum on factor-residual returns is cleaner and far less crash-prone — the
  desk's natural next study (§3.7).

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference.** Newey & West, *Econometrica* 1987 — the WML alpha *t*.
- **Reproducibility.** Headline numbers are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (an as-of date + a content fingerprint of the panel); the cross-section is built with
  [`quantlab.universe`](../../../quantlab/universe.py).

## Caveats stated in the open (house rule)

- **Survivorship bias.** The real panel uses *current* S&P 500 membership, excluding delisted names; the
  qualitative winners-minus-losers ranking is robust, precise magnitudes are not. The long-run premium
  lives in delisted-inclusive, all-cap data (a beat-7 fork).
- **Total-return closes; current large-cap universe.** Momentum is a total-return statement; the modern
  large-cap window is one regime (a growth-led bull with a sharp 2020 reversal), so the `WEAK` stamp
  records the effect's faintness *here*, not a refutation.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
