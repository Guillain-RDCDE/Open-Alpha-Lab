# References & literature map — Study 573 (Customer-Concentration)

## The claim, at full strength

- **Patatoukas (2012)**, *"Customer-Base Concentration: Implications for Firm Performance and
  Capital Markets."* *The Accounting Review* 87(2). The foundational empirical statement: firms with
  a more concentrated customer base show higher operating performance and asset turnover in the
  short run, but bear more **cash-flow and return volatility** — the risk side of the concentration
  story.
- **Dhaliwal, Judd, Serfling & Shaikh (2016)**, *"Customer Concentration Risk and the Cost of Equity
  Capital."* *Journal of Accounting and Economics* 61(1). The pricing claim: major-customer
  concentration raises a supplier's **cost of equity** — evidence that concentration is a *priced*
  risk factor (the premium leg this study tests), driven by cash-flow fragility and lower
  diversification of demand.
- **Hertzel, Li, Officer & Rodgers (2008)**, *"Inter-firm Linkages and the Wealth Effects of
  Financial Distress along the Supply Chain."* *Journal of Financial Economics* 87(2). The
  fragility mechanism: distress propagates *up* the supply chain — a customer's trouble hurts its
  concentrated suppliers — the real-world channel behind concentration risk.
- **Campello & Gao (2017)**, *"Customer Concentration and Loan Contract Terms."* *Journal of
  Financial Economics* 123(1). Lenders price concentration too (higher spreads, more covenants),
  corroborating that concentration is a genuine risk characteristic rather than a labeling artifact.
- **Irvine, Park & Yıldızhan (2016)**, *"Customer-Base Concentration, Profitability, and the
  Relationship Life Cycle."* *The Accounting Review* 91(3). Nuances the sign: concentration can be
  *beneficial* once a supplier–customer relationship matures — one reason the *return* leg's sign is
  genuinely ambiguous (premium vs discount).

## The measure we build

- The literature measure is a **Herfindahl index of revenue shares across a firm's major
  customers** (or the fraction of revenue from customers exceeding the SFAS 131 10%-of-revenue
  disclosure threshold). A firm selling to one dominant customer scores near 1; a firm with many
  small customers scores near 0. This study draws that Herfindahl-style score synthetically (Beta
  right-skew: most firms diversified, a fragile right tail) because the real disclosure data is not
  freely available point-in-time — see the data-availability note below.

## Data-availability limitation (why synthetic-only)

Customer-concentration data comes from 10-K "major customer" footnotes (SFAS 131 segment reporting)
and Compustat's *Segment* files — paywalled, and in most academic work hand-collected or
license-gated. A no-key retail stack cannot assemble a point-in-time concentration panel, so **this
study is synthetic-only** and its Signal axis is capped at `WEAK` (a `REAL` stamp requires a robust
t ≥ 2 on a *real* tape). This mirrors the desk's other no-free-data studies (lego-returns,
whisky-cask, sneaker-resale), which state the data gap on the Signal axis.

## Neighbours on this bench (the dedup map)

- **[Study 540 — Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)** — a *firm-level*
  fundamental-risk sort (bankruptcy probability → returns). Customer concentration is a distinct
  fragility channel (demand-side / supply-chain), not a balance-sheet distress score.
- **[Study 177 — Megacap-Concentration](../../177-megacap-concentration/)** — *index*-level
  concentration (a few mega-caps dominating a benchmark). Study 573 is *firm*-level customer
  concentration (a supplier depending on a few buyers) — the same word, an orthogonal phenomenon.
- **[Study 295 — Stablecoin-Supply](../../295-stablecoin-supply/)** — an unrelated "concentration
  of supply" in crypto; named only to disambiguate the keyword.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the concentrated-minus-diversified
  bucket spreads (both the vol and the return legs).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null for the
  return long-short: shuffle the concentration labels against forward returns and read the spread's
  tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on a *real* tape for `REAL`; synthetic-only ⇒ `WEAK`/`NONE`), seed-robust synthetic controls
  (≥ 20 seeds), one documented execution lag, and costs one-way × NAV with shorts paying borrow.
