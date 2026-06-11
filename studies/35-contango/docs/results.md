# Results — Study 35 (Contango): commodity roll yield, on the real energy tape

> **Real run · offline from cache · as-of 2026-06-05 · inputs fingerprint `92a7674a430b`.**
> Roll yield needs the **term structure** — where on the curve you sit. We observe it without any paid
> futures feed (no FRED, no EIA) by contrasting, for each energy commodity, the **front-month** ETF against
> the **12-month-laddered** one: WTI **USO vs USL**, natural gas **UNG vs UNL**. The laddered fund barely
> touches the front-month roll, so the weekly return spread `laddered − front` is the realized **roll cost of
> the front contract** — positive in contango, negative in backwardation. Both ETFs are liquid with clean
> yfinance history (USL from 2007, UNL from 2010). Reproduce: `python examples/verify.py` (offline, reads
> the cache); refresh the tape with `--fetch`. The cross-sectional bucket machinery is proved on the offline
> synthetic panel — `python examples/run_synthetic_demo.py`.

## The verdict — Signal `WEAK` · Tradability `MIRAGE` · Real-tape run? `DONE`

The commodity carry / roll-yield premium — a long futures position earns (or pays) the **roll** as it slides
along the term structure: backwardation rolls *up* (you bank it), contango rolls *down* (it bleeds you)
(Gorton–Rouwenhorst 2006; Erb–Harvey 2006; Koijen–Moskowitz–Pedersen–Vrugt 2018). On the real energy tape
the roll drag is **economically huge and one-directional** — the most famous wealth-destroyer in retail
commodities, the **USO bleed**, is exactly this number — but on this short, violent, two-curve tape it does
**not clear the desk's statistical bar**: the weekly roll spread's HAC *t* is **+1.53** (WTI) and **+1.75**
(gas), under the house *t* ≥ 2 rule — and our own pre-registered line says HAC *t* < 2 on the real spread
drops the signal to `WEAK`. (The synthetic control's +27.6%/yr spread is a *machinery* proof — the premium
is wired in there by construction — and is not evidence about the market.) Turning the drag into a clean,
tradable carry book on the liquid contracts is a **`MIRAGE`**: the timing signal points the right way yet is
statistically indistinguishable from zero and carries 80%-deep drawdowns. The honest read: contango pointed
one way for two decades and cost the front-month holder ~5–9%/yr, but two noisy curves can't certify it —
the reliable way to "win" is to **not be the sucker holding the front-month**, not to harvest a positive
carry alpha.

## (A) The contango bleed is economically enormous — and statistically under the bar

The realized roll cost of the front-month contract = `laddered − front` return. The front-month funds
bled exactly as the term structure predicts:

| commodity | front | laddered | sample | front total | laddered total | **gap** | roll drag | weeks in contango | HAC *t* |
|---|---|---|---|---|---|---|---|---|---|
| **WTI** | USO | USL | 2007-12 → 2026-06 (18.6y) | **−76%** | **+4%** | **+80 pts** | **+5.1%/yr** | 53% | +1.53 |
| **Natural gas** | UNG | UNL | 2010-01 → 2026-06 (16.5y) | **−99%** | **−88%** | **+11 pts** | **+8.9%/yr** | 56% | +1.75 |

USO lost **three-quarters of its value to the roll** while the laddered USL, on the *same* crude, was flat-to-
positive — an 80-point gap that is pure term-structure cost. Natural gas is worse still: the front-month UNG
is down **−99%**, bleeding **+8.9%/yr** to a curve that sat in contango 56% of all weeks. This is the carry
premium of §9.1 made concrete: backwardation pays, contango taxes, and in energy the tax is brutal. But the
weekly drag's Newey–West *t* is only **+1.5 to +1.8** — under the desk's *t* ≥ 2 bar, and our pre-registered
mirage line (*"real roll spread with HAC t < 2"*) applies to exactly this number. The cumulative direction
points one way and is economically hard to dismiss, but on a 16–19-year weekly sample this volatile, the
spread is not statistically certifiable — which is why the Signal stamp is **`WEAK`**, with the literature's
broad cross-sections (not these two curves) carrying the existence case.

## (B) …but timing it into a carry book is a MIRAGE

The strategy read: hold the front-month ETF only when the curve has recently been **backwardated** (trailing
13-week roll positive), short it in **contango** — a causal, lagged, time-series roll-yield carry book per
curve and equal-weighted across the two.

| book (gross) | Sharpe | CAGR | max-DD | skew | HAC *t* | turnover/yr |
|---|---|---|---|---|---|---|
| WTI | **+0.35** | +6.4% | −76% | +0.71 | +1.45 | 14.7 |
| GAS | +0.04 | −9.3% | −94% | +0.16 | +0.17 | 13.6 |
| **WTI+GAS combo** | **+0.16** | +0.4% | −83% | +0.45 | **+0.66** | 14.1 |
| combo, net @10 bp | +0.12 | −1.1% | −83% | +0.46 | +0.47 | — |

- **The sign is right and it dwarfs the naive trade.** Timing WTI by the curve earns Sharpe **+0.35** with a
  **positive** skew (+0.71) — versus simply holding USO, which earned **−0.01** and lost **−98%** peak-to-
  trough. Knowing the curve turns the bleed into a small gain and dodges the catastrophe. That much is
  directionally right.
- **But as a tradable edge it vanishes.** The combined book's Sharpe is **+0.16** with a Newey–West
  **t = 0.66** — *indistinguishable from zero* — at a **−83% drawdown**. Gas timing earns essentially nothing
  (Sharpe +0.04, *t* +0.17). This trips the pre-registered **mirage line** (`HAC t < 2` on the real spread ⇒
  signal `WEAK`, trade `MIRAGE`) — a line that, read by its own terms, the roll spread in (A) also trips.
- **Cost is not the killer.** The book turns over ~14×/yr but a 10 bp round-trip only nicks the combo
  (+0.16 → +0.12) — the binding constraint is the **crash-prone, two-name concentration** of the only
  energy curves liquid enough to trade, exactly the `FRAGILE`/`MIRAGE` tradability the synthetic control's
  crash tail foretold.
- **One asymmetric cost is not modelled.** The 10 bp covers commissions/spread only: the short legs pay a
  **borrow fee** on USO/UNG (hard-to-borrow in stressed tapes) and the funds' **expense ratios** (~0.6–1%/yr)
  drag the long legs — both unmodelled, both pointing down.

So the roll-yield **force** is economically real-looking but statistically `WEAK` on this tape (the bleed is
one of the largest costs in commodities, yet HAC *t* < 2), and **harvesting** it on the liquid energy tape is
a `MIRAGE`: the timing book is statistically flat and deeply drawdown-prone. The value of the signal is
**defensive** — avoid being long the front-month in contango — not a source of positive carry alpha.

## What the offline synthetic control proves (the machinery)

On a synthetic 12-commodity weekly panel with a *baked* roll-yield premium (seed 35, 20 years,
`carry_strength=0.9`), fingerprint `b502aaa6304f`, the **cross-sectional** bucket book recovers it cleanly:
high-minus-low roll-yield spread **+27.6%/yr**, gross Sharpe **+1.86**, net @5 bp **+1.80**; the disconnected
null (`carry_strength=0`) collapses to Sharpe **−0.28** — the apparatus measures the effect, not itself; and
a carry⊕momentum blend lifts the Sharpe **above either leg** (carry 1.80, momentum 1.43 → blend 2.03) at a
low leg correlation +0.27. **These numbers prove the machinery, not the market** — the premium is baked into
the panel by construction, so they carry no evidential weight for the Signal stamp. The control demonstrates
the bucket machinery *can* harvest carry where a broad cross-section exists; the real energy tape shows that
on the two liquid curves you can actually trade, the cross-section is too thin and the stream too crash-prone
for the harvest to clear noise — `WEAK` force on this tape, `MIRAGE` trade. Reproduce:
`python examples/run_synthetic_demo.py`.

*Sources & literature map: [docs/references.md](references.md); the carry⊕momentum writeup is in
[docs/extension.md](extension.md). Engine: [`quantlab/`](../../../quantlab/). **Not investment advice** —
research & education.*
