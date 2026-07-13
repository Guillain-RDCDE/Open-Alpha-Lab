# References & literature map — Study 717 (Person-of-the-Year)

## The claim under test

- **The folklore — the "magazine cover curse."** The trading-floor belief that when a
  company or its CEO lands a triumphant magazine cover, the peak is in: the stock
  underperforms afterward. TIME's **Person of the Year**, announced every mid-December, is
  the most-watched cover of all, so the believers' one-line question is: *does putting a CEO
  on the Person-of-the-Year cover jinx the stock?* We test whether the honoree's **abnormal**
  return over the following 1–12 months is reliably negative, and whether any decline is a
  *cover* effect or plain **selection** (magazines crown winners at their zenith).
- **TIME's Person of the Year archive.** The primary source for the honoree list and the
  mid-December announcement dates: TIME, *Person of the Year* (`time.com/person-of-the-year`),
  and the contemporaneous cover issues (Jeff Bezos, 1999; "The Good Samaritans" incl. Bill
  Gates, 2005; Mark Zuckerberg, 2010; Elon Musk, 2021; Donald Trump, 2016 & 2024). Only
  honorees with a public stock **at** the announcement enter the tradable table; the rest are
  named on the Signal axis.

## The cover-curse / superstar literature

- **Arnott, Wu & Chaves — the underlying mechanism.** Rob Arnott's work on the mean-reversion
  of glamour/growth and on "superstar" firms: extreme prior winners systematically
  disappoint. The cover is a *symptom* of the extension, not a cause — precisely the
  misattribution our run-up regression isolates.
- **Malmendier & Tate (2009), *Superstar CEOs* (Quarterly Journal of Economics).** Award-
  winning CEOs (a cousin of the cover) subsequently **underperform** — but the paper's own
  reading is agency/distraction and, crucially, *selection*: award-winners were already at a
  performance peak. The canonical academic statement of the effect we test and then attribute
  to selection.
- **The "Sports Illustrated / BusinessWeek cover jinx" folklore.** The popular-press ancestor
  of the claim (e.g. the recurring "cover-story curse" columns) — universally explained by
  **regression to the mean** once anyone measures it, never by a magazine having causal power.

## The event-study method (the shared engine)

- **Market model / abnormal returns.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment of
  Stock Prices to New Information* (Int. Econ. Review) — the original event study. Brown &
  Warner (1985), *Using daily stock returns: The case of event studies* (J. Financial
  Economics) — the canonical daily-data methodology we follow: estimate `r_stock = α + β·r_mkt`
  on a clean pre-event window, then cumulate AR = r_stock − (α + β·r_mkt) over the event
  window. MacKinlay (1997), *Event Studies in Economics and Finance* (J. Economic Literature)
  — the textbook synthesis (estimation window, event window, CAR).
- **Long-horizon caveats.** Barber & Lyon (1997) and Kothari & Warner (1997) — long-horizon
  abnormal-return tests are badly mis-specified and low-powered; CAR (additive) and BHAR
  (compounded) diverge, and small samples of extreme names inflate apparent significance.
  Directly relevant: our *t* = −2.95 comes from **four** heavy-tailed names.

## Why four events + selection cannot certify a curse — the statistics

- **Small-sample inference / power.** With `k = 4` events, the standard error of a mean CAR is
  large and dominated by the two bubble-era icons. We test the pooled mean against zero with a
  **Welch t** (Welch, 1947), and — because `k` is tiny and CARs are heavy-tailed — against a
  **placebo / randomization null**: random non-event mid-December windows on the same names
  (Fisher's randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Selection / momentum confound.** Jegadeesh & Titman (1993) momentum and its violent
  **crashes/reversals** (Daniel & Moskowitz, 2016, *Momentum crashes*, JFE): extreme prior
  winners revert. Regressing post-CAR on the **prior-year run-up** (a one-factor selection
  control) drives the residual curse to zero — the effect is momentum mean-reversion, not a
  cover. Window/selection cautions: Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns*; Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.

## Method lineage (the desk's shared engine)

- **Market-model long-horizon CAR + prior run-up.**
  [`strategy.event_car`](../person_of_the_year/strategy.py),
  [`strategy.car_panel`](../person_of_the_year/strategy.py) and
  [`strategy.prior_runup`](../person_of_the_year/strategy.py) — abnormal returns cumulated over
  the post-coronation window, plus the selection control.
- **Welch t + placebo p-value.** [`strategy.welch_t`](../person_of_the_year/strategy.py) and
  [`strategy.placebo_car_dist`](../person_of_the_year/strategy.py) — the Signal-axis tests:
  pooled mean vs zero and a random mid-December-window null.
- **Borrow-aware short costs.** [`strategy.net_of_costs`](../person_of_the_year/strategy.py) —
  the curse trade is a short; gross and net of stock borrow + one round-trip.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../person_of_the_year/data.py) plants a known post-coronation
  drift; the offline core runs with no network. The control confirms the engine recovers a
  planted curse **and** that four heavy-tailed events cannot reach significance unless the
  planted edge is large.

## Data sources used here

- **Hardcoded honoree table** (`person_of_the_year.data.POY_EVENTS`): the TIME Person-of-the-
  Year honorees who ran or were the face of a public company at the announcement (AMZN'99,
  MSFT'05, TSLA'21, DJT'24), with mid-December announcement dates from TIME's archive. True
  cover-effect studies would need every business cover ever printed; the dated, labelled
  Person-of-the-Year census is the transparent stand-in.
- **yfinance** daily adjusted (total-return) closes for each honoree ticker + SPY, cached under
  `_cache/`. Two business honorees (META'10 Zuckerberg, DJT'16 Trump) drop out for lack of a
  public stock at the cover — a survivorship note on the Signal axis. All headline numbers are
  pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the same market-model event study on a
  different dated corporate catalyst — there the killer is that the move is priced before you
  can act; here it is that the move is *selection*, not the event.
- **[Study 344 — Backtest-Overfitting](../344-backtest-overfitting/)** and the run-up control
  here share one lesson: an effect that clears *t* = 2 raw can dissolve entirely under the one
  obvious confound.
