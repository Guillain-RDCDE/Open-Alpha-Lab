# References & literature map — Study 544 (Oyster-R-Months)

## The claim, at full strength (the folk rule)

- **The "R-month" oyster rule.** English-language folk wisdom, at least as old as William Butler's
  *Dyets Dry Dinner* (1599): *"It is unseasonable and unwholesome in all months that have not an R
  in their name to eat an oyster."* The R-less months are **May, June, July, August**; the R-months
  are **September–April**. The rationale is biological and pre-refrigeration: oysters spawn in the
  warm summer months (turning soft and milky) and shellfish spoils fastest in summer heat. It is a
  food-safety heuristic, not a market claim — this study asks, tongue-in-cheek, whether markets
  happen to honour the same calendar.

## The market seasonal it actually maps onto

- **Bouman & Jacobsen (2002)**, *"The Halloween Indicator, 'Sell in May and Go Away': Another
  Puzzle."* *American Economic Review* 92(5). The canonical documentation of the **sell-in-May /
  Halloween** seasonal: equities earn more Nov–Apr ("winter") than May–Oct ("summer") across many
  markets. The R-month split (Sep–Apr vs May–Aug) is calendrically almost identical — one month
  wider on the winter side (it moves **September** to the hold side) — so the oyster rule is, on the
  tape, a *variant* of sell-in-May.
- **Jacobsen & Zhang (2018)**, *"The Halloween Indicator: Everywhere and All the Time."* The
  large-sample update confirming the winter-summer gap is widespread and long-lived, while noting
  its softness and the difficulty of trading it net of the positive summer drift.
- **Bouman & Jacobsen critics** (e.g. **Maberly & Pierce 2004**) — the seasonal is sensitive to a
  handful of outlier months and sub-periods; a reminder to read the *t*, the placebo and the
  sub-periods, not just the point estimate.

## Neighbours on this bench (the dedup map)

- **[Study 55 — Summer-Lull](../../55-summer-lull/)** — the direct sell-in-May / Halloween study on
  98 years of the S&P 500. Study 544 is the *oyster-rule costume* of that seasonal: it uses the
  R/non-R month split (which adds September to the hold side) and adds a **consumer-staples**
  instrument, precisely because the oyster rule is about *seafood/food*. The headline finding here
  is that the R-month framing is a **worse** sell-in-May (September is the market's worst month).
- **[Study 223 — Same-Month-Seasonality](../../223-same-month-seasonality/)** and
  **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)** — other calendar effects; this
  study is specifically the R-month / sell-in-May folk split, not a per-month or intra-month effect.
- **[Study 95 — Holiday-Cheer](../../95-holiday-cheer/)** / **[Study 158 — Super-Bowl](../../158-super-bowl/)**
  — the "folklore & spurious calendar" family this study belongs to.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the R-month vs R-less monthly-return
  gap and for the Halloween cousin.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  R/non-R month labels against the returns and read the gap's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on the real tape for `REAL`, plus a placebo null and a seed-robust synthetic control), gross/net
  labelling, one documented execution convention, and shorts paying borrow.
