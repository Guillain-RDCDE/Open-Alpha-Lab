# References & literature map — Study 520 (External-Financing-Anomaly)

## The claim, at full strength

- **Bradshaw, Richardson & Sloan (2006)**, *"The Relation Between Corporate Financing Activities,
  Analysts' Forecasts and Stock Returns."* *Journal of Accounting and Economics* 42(1–2). The
  source paper: a firm's **net external financing** (the cash raised from debt *and* equity, net of
  what is returned), taken straight from the cash-flow statement and scaled by assets, predicts
  *lower* subsequent stock returns and over-optimistic analyst forecasts. The signal this study
  replicates.
- **Cooper, Gulen & Schill (2008)**, *"Asset Growth and the Cross-Section of Stock Returns."*
  *Journal of Finance* 63(4). The investment side of the same coin — total-asset growth predicts low
  returns — overlapping with where the raised cash goes. (This desk's [Study 244](../../244-asset-growth/).)
- **Daniel & Titman (2006)**, *"Market Reactions to Tangible and Intangible Information."* *Journal
  of Finance* 61(4). Net-issuance / financing as the "tangible" composite-issuance signal that
  forecasts returns.
- **Pontiff & Woodgate (2008)**, *"Share Issuance and Cross-Sectional Returns."* *Journal of Finance*
  63(2). The equity-issuance leg of external financing as a standalone predictor.
- **Loughran & Ritter (1995)**, *"The New Issues Puzzle."* *Journal of Finance* 50(1). The original
  underperformance-after-issuance evidence (SEOs/IPOs) that the financing anomaly generalises.

## Why the honest replication tends to disappoint

- **McLean & Pontiff (2016)**, *"Does Academic Research Destroy Stock Return Predictability?"*
  *Journal of Finance* 71(1). Post-publication decay: anomalies shrink ~58% out of sample — the
  external-financing effect among them.
- **Survivorship bias**: the basket is names still trading in 2026. The dead raisers (the firms the
  anomaly says should have cratered) are absent, which can only *flatter* the short (raiser) leg —
  stated openly on the SIGNAL axis.

## Neighbours on this bench (the dedup map)

- **[Study 64 — Share-Shuffle](../../64-share-shuffle/)** — the *net share-issuance* anomaly (equity
  only). Study 520 is the *combined* debt **and** equity external-financing aggregate off the
  cash-flow statement, not equity issuance alone.
- **[Study 244 — Asset-Growth](../../244-asset-growth/)** — total-asset growth (the *uses* of the cash).
  Study 520 sorts on the *financing* side (the *sources*).
- **[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/)** / **[231 — Sloan Accruals](../../231-sloan-accruals/)**
  / **[52 — Smoke-Screen](../../52-smoke-screen/)** — balance-sheet bloat and accruals (the quality of
  earnings). Study 520 is the cash-flow-statement *financing-activities* aggregate, a distinct line.
- **[Study 368 — Buyback-Drift](../../368-buyback-drift/)** — the *announcement* drift after a buyback.
  Study 520 uses the realised *financing flow* (issuance net of repurchase), not the announcement.

## Shared method

- **Newey & West (1987)** — heteroskedasticity- and autocorrelation-consistent (HAC) standard
  errors; the *t* the inference bar requires.
- House methodology: [`METHODOLOGY.md`](../../METHODOLOGY.md) — the inference bar, the label-shuffle
  placebo null, one execution lag, costs one-way × NAV with shorts paying borrow, the synthetic
  control as a power proof only.
