# References & literature map — Study 542 (Pi-Day-Effect)

## The claim, such as it is

Pi Day (March 14) is a cultural observance, not an academic finance claim — which is precisely the
point of this study. It stands in for the whole genre of **numerology / "special date" anomalies**:
the idea that a calendar coincidence (a date that spells out π, or the digits of *e*) carries market
information. There is no peer-reviewed paper claiming a Pi-Day return premium; the honest test is
therefore whether the constant dates are more extreme than a *random* set of dates — the null this
study is built around.

## The genuine calendar-anomaly literature it satirises

- **Kolb & Rodriguez (1987)**, *"Friday the Thirteenth: 'Part VII' — A Note."* *Journal of Finance*
  42(5). A negative Friday-13th anomaly in the 1940-1987 DJIA — the archetypal "superstition moves
  the tape" finding, and the direct cousin of this study ([Study 163](../../163-friday-13th/)).
- **Lakonishok & Smidt (1988)**, *"Are Seasonal Anomalies Real? A Ninety-Year Perspective."*
  *Review of Financial Studies* 1(4). The definitive audit of calendar effects (turn-of-month,
  weekend, holiday, January): several are fragile, most shrink out of sample.
- **Ariel (1990)**, *"High Stock Returns Before Holidays."* *Journal of Finance* 45(5). A real
  pre-holiday effect — the kind of calendar signal a numerology date is *not*.
- **Sullivan, Timmermann & White (2001)**, *"Dangers of Data Mining: The Case of Calendar Effects
  in Stock Returns."* *Journal of Econometrics* 105(1). The core warning: test enough calendar
  rules and one will look significant by chance. The multiple-testing rationale for this study's
  random-date-set placebo and Bonferroni sweep.

## Why the placebo is the whole game

- **Bonferroni (1936)** / **Šidák (1967)** — family-wise error correction across the six constant
  dates: choosing dates from the calendar is a multiple-comparisons problem, and the correction
  prices it in.
- **Permutation / randomisation testing** (Fisher 1935; Good 2005) — the random-date-set null:
  resample sets of *K* calendar slots and read the observed contrast's tail probability. This is
  the honest way to ask "is π special, or would any six dates do this well?"
- **Welch (1947)** — the unequal-variance two-sample *t* used for the constant-day vs rest contrast.
- **Newey & West (1987)** — the HAC standard error on the single-group mean (autocorrelation-robust
  inference, per the desk's inference bar).

## Neighbours on this bench (the dedup map)

- **[Study 163 — Friday-13th](../../163-friday-13th/)** — the closest cousin: a superstition/date
  anomaly with a placebo and Bonferroni sweep. That study tests a *weekday×day-of-month* legend
  (the 13th vs other Fridays); this one tests **mathematical-constant month/day dates** (π, *e*, τ,
  φ, √2, δ) with a **random-date-set** null.
- **[Study 285 — St-Patrick's-Day](../../285-st-patricks-day/)** / **[Study 286 —
  Valentine's-Day](../../286-valentines-day/)** — single-holiday date effects. Same family; this
  study pools six *numerology* dates and uses the random-date-set placebo rather than a single
  holiday.
- **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)**, **[Study 90 —
  Weekend](../../90-weekend/)**, **[Study 95 — Holiday-Cheer](../../95-holiday-cheer/)**,
  **[Study 224 — Monday-Effect](../../224-monday-effect/)** — the *real* calendar anomalies the
  numerology dates are contrasted against.

## Shared method

- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (HAC *t*,
  a placebo null, seed-robust synthetic control), gross-and-net costs, and the honest naming of
  the thin-event-count power limitation on the SIGNAL axis.
