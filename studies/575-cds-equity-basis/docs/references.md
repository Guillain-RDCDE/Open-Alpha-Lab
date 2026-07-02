# References & literature map — Study 575 (CDS-Equity-Basis)

## The claim, at full strength

- **Kapadia & Pu (2012)**, *"Limited Arbitrage Between Equity and Credit Markets."* *Journal of
  Financial Economics* 105(3). The canonical study of the **CDS-equity basis**: the two markets
  are *co-integrated* but converge only slowly and imperfectly, precisely because arbitrage between
  them is *limited* (funding, illiquidity, idiosyncratic risk). The convergence is statistically
  present but economically faint — the empirical anchor for why this effect is *weak*, not free.
- **Merton (1974)**, *"On the Pricing of Corporate Debt: The Risk Structure of Interest Rates."*
  *Journal of Finance* 29(2). The structural model that turns a firm's equity price and equity
  volatility into an *implied credit spread* — the ``eq_impl_bp`` leg of the basis. Equity and
  credit are contingent claims on the same firm value, so a dislocation between them is an
  arbitrage-flavoured signal.
- **Duffie (1999)**, *"Credit Swap Valuation."* *Financial Analysts Journal* 55(1). The reduced-form
  pricing of the CDS spread itself — the ``cds_bp`` leg — and the arbitrage relation that ties a
  CDS spread to a bond/equity-implied credit level.
- **Bai & Collin-Dufresne (2019)**, *"The CDS-Bond Basis."* *Financial Management* 48(2). The
  sibling *CDS-bond* basis and its persistence through the crisis — evidence that credit-market
  bases are driven by funding and limits-to-arbitrage, not free money. A neighbour construct that
  frames why the *equity* basis is likewise hard to harvest.
- **Acharya & Johnson (2007)**, *"Insider Trading in Credit Derivatives."* *Journal of Financial
  Economics* 84(1). Evidence that the CDS market can *lead* the equity market for distressed names —
  the lead-lag mechanism the convergence trade tries to monetise.

## The measure we build

- The **basis** is ``s - c``: the single-name CDS spread ``s`` minus a Merton-style
  equity-implied spread ``c``, both in bp. A positive basis means credit is pricing *more* distress
  than equity. The synthetic panel simulates two coupled AR(1) risk views whose transient
  dislocation *is* the basis, and lets the *next* month's equity return load on the *current* basis
  with a tunable ``convergence_beta`` (negative = the folklore convergence sign). The real
  equivalent needs licensed single-name CDS (Markit/IHS) plus point-in-time liabilities — hence
  the study is synthetic-only and the data gap is named on the SIGNAL axis.

## Neighbours on this bench (the dedup map)

- **[Study 382 — Treasury-Basis-Trade](../../382-treasury-basis-trade/)** — a *rates* funding
  basis (cash-vs-futures carry / implied repo), a term-premium story. Study 575 is a *cross-asset
  credit-vs-equity* dislocation, a lead-lag prediction of equity returns — different market,
  different mechanism.
- **[Study 123 — Altman-Z](../../123-altman-z/)** / **[Study 230 — Ohlson-O-score](../../230-ohlson-o-score/)**
  / **[Study 540 — Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)** — *equity-only*
  distress scores and the distress return puzzle. Study 575 adds the *credit market's* view (CDS)
  as a second lens and trades the *gap* between the two, not a one-market distress score.

## Shared method

- **Cluster-robust (by month) standard errors** (Liang & Zeger 1986; Cameron & Miller 2015) — the
  pooled panel regression clusters by month so the within-month cross-sectional correlation cannot
  inflate the slope *t*.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  basis labels against forward returns within each month and read the long-short's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a **real** tape for REAL; a synthetic-only study is capped at WEAK/NONE), one
  execution lag (the one-month basis→forward-return lag), gross/net labelled, shorts paying borrow,
  and the seed-robust (≥ 20 seeds) synthetic positive control.
