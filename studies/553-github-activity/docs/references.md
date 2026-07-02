# References & literature map — Study 553 (GitHub-Activity)

## The claim, at full strength

- **The alt-data thesis.** A firm's *observable engineering telemetry* — public commit cadence,
  merged pull-requests, contributor growth, repository stars — is a real-time proxy for
  **innovation intensity** and product-shipping velocity, and therefore a *nowcast* of
  fundamentals the market has not yet priced. This is the software-native cousin of the classic
  "alternative-data" playbook (satellite car-counts, credit-card panels, web traffic) applied to
  a company's own open-source output.
- **GitHub as the substrate.** GitHub's own *Octoverse* reports and the public GitHub Archive
  (githubarchive.org / the GH Archive on BigQuery) document firm and ecosystem activity at scale —
  the raw material anyone imagining this signal would reach for.

## Why "innovation intensity predicts returns" is a *real* literature (the steelman)

- **Chan, Lakonishok & Sougiannis (2001)**, *"The Stock Market Valuation of Research and
  Development Expenditures."* *Journal of Finance* 56(6). R&D-intensive firms earn higher
  subsequent returns — the market under-weights intangibles. The economic prior behind
  "measure innovation, predict returns."
- **Hirshleifer, Hsu & Li (2013)**, *"Innovative Efficiency and Stock Returns."* *Journal of
  Financial Economics* 107(3). Patents/citations *per R&D dollar* forecast returns and profits —
  a distinct, output-based innovation signal (the spirit this study's commit-velocity proxy
  imitates for software firms).
- **Cohen, Diether & Malloy (2013)**, *"Misvaluing Innovation."* *Review of Financial Studies*
  26(3). The market misprices the *ability* of firms to turn R&D into products — the
  under-reaction the "read the factory floor early" alt-data pitch is really betting on.

## The alt-data reality check (why the ceiling is WEAK/MIRAGE)

- **Green, Hand & Zhang (2013)** and the broader factor-zoo literature — most freshly proposed
  cross-sectional predictors carry *small* information-coefficients that do not survive out of
  sample; a realistic alt-data IC of ~0.03–0.05 needs a long, clean tape to certify.
- **Survivorship & point-in-time integrity.** The decisive practical obstacle: GitHub feeds are
  *current* snapshots (renamed/archived/privatised repos vanish; histories are rewritable) and
  rate-limited, so a retail stack cannot reconstruct the velocity known at each *past* date. Any
  free backtest is look-ahead- and survivorship-contaminated — which is exactly why this study is
  synthetic-only and states the limitation on the SIGNAL axis.

## Neighbours on this bench (the dedup map)

- **[Study 400 — Patent-Intensity](../../400-patent-intensity/)** — the *audited* innovation
  proxy (SEC R&D / revenue) and its long-short. Study 553 is the **public open-source telemetry**
  cousin: a cross-sectional *IC* nowcast rather than an R&D-intensity tertile trade, and
  synthetic-only because no point-in-time GitHub tape exists.
- **[Study 334 — ARK-Innovation](../../334-ark-innovation/)** — "innovation" as a *product/theme*
  basket; here innovation is a *firm-level telemetry z-score*.
- **[Study 392 — Glassdoor-Sentiment](../../392-glassdoor-sentiment/)** /
  **[Study 528 — Labor-Hiring-Rate](../../528-labor-hiring-rate/)** — other *firm-telemetry*
  alt-data nowcasts (employee sentiment, hiring). Study 553 is the **engineering-output** telemetry
  variant, and shares their central caveat: the free feed is not point-in-time.
- **[Study 335 — Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** — social-media buzz as
  alt-data; commit/star velocity is the *builder-side* analogue of the *audience-side* buzz signal.

## Shared method

- **Cross-sectional information coefficient (IC).** The per-period Spearman rank correlation
  between the signal and the forward return; the headline test is the mean IC's *t*-stat over
  periods (Grinold & Kahn, *Active Portfolio Management*, 1999 — the "fundamental law").
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle
  velocity against forward returns *within each period* and read the mean-IC tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 plus a placebo null and seed-robustness), the explicit data-availability caveat on the
  SIGNAL axis, one execution lag, and costs one-way × NAV with shorts paying borrow.
