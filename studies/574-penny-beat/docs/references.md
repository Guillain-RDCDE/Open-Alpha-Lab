# References & literature map — Study 574 (Penny-Beat)

## The claim, at full strength — the earnings discontinuity

- **Burgstahler & Dichev (1997)**, *"Earnings Management to Avoid Earnings Decreases and Losses."*
  *Journal of Accounting and Economics* 24(1). The foundational discontinuity: an *unusually low*
  frequency of small losses / small earnings decreases and an *unusually high* frequency of small
  profits / small increases — mass shifted across the zero threshold. The histogram-kink logic this
  study proxies for the analyst-consensus threshold.
- **Degeorge, Patel & Zeckhauser (1999)**, *"Earnings Management to Exceed Thresholds."* *Journal of
  Business* 72(1). Formalises the three thresholds firms manage to — positive profits, last year's
  earnings, and the **analyst consensus** — and documents the discontinuity at each. The consensus
  threshold is exactly the penny-beat: firms nudge a small miss into a small beat.
- **Matsumoto (2002)**, *"Management's Incentives to Avoid Negative Earnings Surprises."* *The
  Accounting Review* 77(3). Both real earnings management *and* expectations management (walking the
  consensus down) produce the just-meet-or-beat pattern — the two mechanisms behind the +$0.01 spike.
- **Bhojraj, Hribar, Picconi & McInnis (2009)**, *"Making Sense of Cents: An Examination of Firms
  That Marginally Miss or Beat Analyst Forecasts."* *Journal of Finance* 64(5). The trading-relevant
  paper: firms that *just beat* forecasts (by managing) enjoy a short-term price bump but
  **underperform over the following three years** — the return-penalty claim this study tests.

## The return-penalty debate (why the Signal is `WEAK`, not `REAL`)

- **Ball & Shivakumar (2008)** and later work question whether the discontinuity itself is partly a
  research-design artefact (deflation/rounding, sample selection) rather than pure management — a
  reminder that the *shape* is robust but its interpretation is contested.
- **Post-earnings-announcement drift (PEAD)** — Bernard & Thomas (1989, 1990) — the confound this
  study makes explicit: a +1c surprise is a *smaller* surprise than a decisive ≥+3c beat, so it
  drifts less **even absent any management**. The naive penny-minus-decisive spread conflates the
  management penalty with this mechanical composition gap; only a within-threshold (managed vs
  honest) test isolates the clean effect.

## Neighbours on this bench (the dedup map)

- **[Study 229 — Beneish M-score](../../229-beneish-m-score/)** — the 8-ratio *accounting*
  manipulation composite (DSRI, GMI, TATA…). A firm-level ratio screen; Study 574 is the
  **earnings-surprise histogram discontinuity** (a population *shape*, not a ratio) and its return
  consequence.
- **[Study 328 — Benford-Law](../../328-benford-law/)** — the leading-digit fraud screen on reported
  figures. Also a distributional-anomaly detector, but on *digits*; Study 574 is the +$0.01 spike in
  the *surprise* distribution.
- **[Study 11 — Vanishing-Penny](../../11-vanishing-penny/)** — a *different* penny: a Polymarket
  YES+NO arbitrage. No relation beyond the word.
- Synthetic-only cousins (no free real tape, ceiling `WEAK`): **[273 Lego-Returns](../../273-lego-returns/)**,
  **[275 Whisky-Cask](../../275-whisky-cask/)**, **[276 Sneaker-Resale](../../276-sneaker-resale/)**.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the penny − decisive and the
  within-bin managed − honest spreads.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo nulls: for the
  discontinuity, redraw the local histogram from a smooth shape and read the +1c z's tail; for the
  return spread, shuffle the bucket labels against forward returns.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 **on a real tape** for `REAL`; synthetic-only caps at `WEAK`), one execution lag, costs
  one-way × NAV with shorts paying borrow, and the seed-robust synthetic positive control (≥ 20
  seeds).
