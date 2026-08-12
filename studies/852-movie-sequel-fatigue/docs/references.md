# References & literature map — Study 852 (Movie-Sequel Fatigue)

## The claim under test

- **The folklore.** "Franchise fatigue": as a series grinds out sequel after sequel, each
  new entry is supposed to open weaker than the last — audiences tire, reviews sour, the
  brand wears thin — and the reflex corollary on the stock is that the *studio* should react
  worse to sequel N than to sequel N-1, with a run of tired sequels (a "down sequence")
  hanging over the next entry. The 2022-2024 stretch (*Ant-Man: Quantumania*, *The Marvels*,
  *Indiana Jones 5*, *Fast X*) made "superhero/franchise fatigue" a financial-press staple.
- **Why it's a clean calendar test.** A wide release opens on a known Friday; the weekend
  box-office estimate is public by Sunday and actuals by Monday, so the first trading session
  at which "how it opened" is common knowledge is the **Monday after the opening weekend** —
  anchoring the studio-reaction window there (with the opening-Friday close as the base) is
  calendar-known and zero-look-ahead. Opening dates are hardcoded from Box Office Mojo /
  studio press releases ([`data.py`](../sequel_fatigue/data.py)). The `seq` field is the
  film's true ordinal within its franchise line.
- **The efficient-markets prior.** One film — even a $250M tentpole — is a rounding error for
  a ~$100-200bn conglomerate dominated by parks, streaming, cable and consumer products, and
  the opening is a *scheduled, public* event. Semi-strong efficiency (Fama, 1970, *Efficient
  Capital Markets*, **Journal of Finance**) says it should already be priced — so the honest
  prior is a non-event, and with ~40 events this is a low-power cross-event test.

## What the literature actually says

- **Event-study method** — Brown & Warner (1985, **Journal of Financial Economics**, "Using
  daily stock returns"); MacKinlay (1997, **Journal of Economic Literature**, "Event Studies
  in Economics and Finance"). The canonical abnormal-return / CAR machinery and its
  small-sample cautions; our per-event studio abnormal return (`studio − SPY`) and the
  cross-event *t* follow this template directly.
- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, **JAR**); Bernard &
  Thomas (1989, **JAR**; 1990, **JAE**). The "prices drift after a scheduled information
  event" intuition the "sell the tired sequel" reflex borrows; a box-office weekend is a
  product-news event, not an earnings print, but the drift-after-news framing is the same.
- **Product-market news and firm value** — Chen, Da & Zhao (2013, **RFS**, "What Drives Stock
  Price Movements?") on cash-flow news, and Joshi & Hanssens (2009, *Journal of Marketing*)
  on advertising / box-office effects on firm value — both emphasise that a single product is
  a small share of a diversified firm's cash flows, the core reason the prior is skeptical.
- **Sequels, brand extension and demand** — the marketing/entertainment literature on
  sequels (e.g. Basuroy & Chatterjee, 2008, *Journal of Business Research*, on sequel
  box-office; Hennig-Thurau et al. on movie brand extensions) documents *declining
  box-office* for later sequels — the demand-side "fatigue" — but says little about whether
  that reaches a *diversified distributor's equity* on the opening.
- **Attention & sentiment** — Barber & Odean (2008, **RFS**) on attention-driven trading and
  Da, Engelberg & Gao (2011, **JF**) on search-based attention: a high-profile flop draws
  retail attention and a possible over-reaction, but attention is not itself a tradable edge.
- **Sibling desk studies (dedup).** This study is the *sequel-number / fatigue-slope* cut and
  is deliberately distinct from:
  [771-box-office-bomb](../../771-box-office-bomb/) — "sell Disney after a *single* notorious
  flop" (a per-film bomb, not the sequel-number trend);
  [847-rotten-tomatoes-studio](../../847-rotten-tomatoes-studio/) — the *critic-score*
  (Rotten Tomatoes) reaction, not the sequel ordinal;
  [550-box-office-momentum](../../550-box-office-momentum/) — *momentum* in studio returns
  around box-office, not the within-franchise fatigue tilt.

## Data & method

- **Real tape:** `DIS`, `CMCSA`, `PARA` and `SPY` daily total-return (`auto_adjust=True`)
  closes via [yfinance](https://github.com/ranaroussi/yfinance), cached under this study's own
  `_cache/`. `PARA` (Paramount Global) has continuous adjusted history only from 2021-02, so
  the three pre-2021 Transformers entries are dropped, not back-filled — named honestly.
- **Statistics:** per-event studio abnormal return over `[anchor−1 .. anchor+3]`; one-sample
  and Newey-West(4) *t* on the mean; an OLS **fatigue slope** of CAR on sequel number (raw and
  franchise-fixed-effect); a two-era robustness cut; a within-franchise AR(1) persistence and
  a down/up Welch split; a 5,000-draw sequel-number label-permutation placebo; a 20-seed ×
  200-draw random-date placebo; a costed short-the-fatigued-sequel timer.
- **Synthetic positive control:** a seeded paired (studio, benchmark) price world with a
  *planted* fatigue slope (reaction declines with sequel number) and an optional persistence
  (AR(1)); the detector must recover the plant monotonically and stay quiet on the null
  ([`strategy.py`](../sequel_fatigue/strategy.py)).

*Fama, E. (1970). **Journal of Finance**. · Brown, S. & Warner, J. (1985). **JFE**. ·
MacKinlay, A.C. (1997). **JEL**. · Ball, R. & Brown, P. (1968). **JAR**. · Bernard, V. &
Thomas, J. (1989, 1990). **JAR / JAE**. · Chen, L., Da, Z. & Zhao, X. (2013). **RFS**. ·
Joshi, A. & Hanssens, D. (2009). **Journal of Marketing**. · Basuroy, S. & Chatterjee, S.
(2008). **JBR**. · Barber, B. & Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P.
(2011). **JF**.*
