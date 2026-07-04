# References & literature map — Study 617 (Crash-Insurance-Cost)

## The claim under test

- **The folk version.** "Tail-risk funds and put-buying strategies bleed money every year — the
  crash insurance never pays for itself." Heard after every TAIL drawdown chart on FinTwit; the
  serious version is that the bleed **is the volatility risk premium**, read from the *buyer's*
  side of the trade.
- **The product.** **Cambria Tail Risk ETF (TAIL)**, inception 2017-04-05, expense ratio
  0.59%/yr: ~90% intermediate US Treasuries + a laddered budget (~1%/quarter) of out-of-the-money
  SPX puts. Fund page: https://cambriafunds.com/tail — the design is *public*, which is what lets
  us decompose the fund against its own collateral (IEF).

## The premium the buyer is paying

- **Bakshi & Kapadia (2003),** *Delta-Hedged Gains and the Negative Market Volatility Risk
  Premium*, **RFS 16(2)** — delta-hedged option buyers systematically lose: the direct evidence
  that index options embed a negative volatility risk premium (the buyer pays it).
- **Carr & Wu (2009),** *Variance Risk Premiums*, **RFS 22(3)** — synthetic variance swaps from
  option surfaces: realized variance is persistently *below* the implied (swap) rate; sellers of
  variance are paid, buyers pay. Our `RV − IV` monthly series is exactly their payoff sign
  convention, with VIX² as the swap rate.
- **Coval & Shumway (2001),** *Expected Option Returns*, **JF 56(3)** — zero-beta straddles earn
  significantly negative returns: buying optionality loses on average.
- **Bondarenko (2003),** *Why Are Put Options So Expensive?* (working paper / AFA) — deep OTM
  index puts earn strongly negative average returns even accounting for crash risk; the classic
  "puts are the most expensive insurance in finance" result.
- **Israelov (2019),** *Pathetic Protection: The Elusive Benefits of Protective Puts* (AQR /
  Journal of Alternative Investments) — systematic put protection drags CAGR and barely improves
  drawdowns per unit of cost vs simply de-risking; the direct practitioner version of our
  SPY/TAIL blend table.
- **Dew-Becker, Giglio, Le & Rodriguez (2017),** *The Price of Variance Risk*, **JFE 123(2)** —
  the premium is concentrated in the *front* of the curve, precisely where insurance buyers roll.

## What we measure, and why the decomposition matters

- **Raw drift is not enough.** TAIL is ~90% Treasuries, so its NAV also carries duration —
  2022's bond bear would smear a pure-drift reading. Regressing TAIL on **IEF** (`tail = α +
  β·IEF`) isolates the **put sleeve plus fee** in α: the insurance bill itself. We report daily
  and monthly HAC/Newey-West *t* (Newey & West 1987), the monthly stat as headline (daily
  microstructure noise dilutes *t*; compounding to months restores it), and both crash-free
  sub-periods.
- **The variance-notional series is model arithmetic, labeled.** `IV = (VIX/100)²/12` uses the
  prior month-end VIX (CBOE VIX White Paper: VIX² is the 30-day variance-swap rate) — one clean
  one-month lag; `RV = Σ (daily SPY log return)²`. It is not an option backtest (no strikes, no
  roll mechanics) — it *names the mechanism* the TAIL alpha embodies. Whaley (2009), *Understanding
  the VIX*, JPM, for the index construction.
- **Cohort accounting for the third axis.** Every month-end entry, held to as-of, total-return —
  the honest version of "but 2020 paid": *did anyone keep it?*
- **One lag, documented.** Buy-and-hold TAIL is static (no signal, no lag to apply); the variance
  premium uses strictly prior-month information. Costs: 5 bps one-way × turnover on the blend
  rebalances; TAIL's 0.59%/yr ER is inside its NAV (all TAIL numbers are net of it).

## Data sources used here

- **yfinance** daily closes — TAIL, IEF, SPY (auto-adjusted, total-return) and ^VIX (level),
  1990-01-02 → 2026-06-30, cached under `_cache/cic_tape.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
- **Cambria fund page / prospectus** for TAIL's design facts (inception, ER, put-ladder budget):
  https://cambriafunds.com/tail
- **CBOE VIX White Paper** (variance-swap interpretation of VIX²):
  https://cdn.cboe.com/resources/vix/vixwhite.pdf

## Related desk studies (the other side of this exact trade)

- **[92-easy-money](../92-easy-money/)** — SELLING vol via short VIXY: the contango carry is
  **Real** (HAC *t* = +2.31), Fragile to hold. The premium we watch TAIL *pay* is the one 92
  harvests.
- **[63-free-fall](../63-free-fall/)** — SVXY short-vol carry: **Real**, Fragile (−95% drawdown
  risk). Same premium, packaged short-vol product.
- **[86-tail-radar](../86-tail-radar/)** — the CBOE **SKEW** index as a crash *signal*: **None /
  Mirage**. Distinct axis: 86 asks whether the price of tails *predicts* crashes; this study asks
  what *paying* that price costs a live buyer, year after year.

This study is framed distinctly from all three: not a vol-selling backtest, not a signal test —
a **cost accounting of the buyer's side on a live, buyable product**, decomposed against its own
design.
