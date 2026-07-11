# References & literature map — Study 659 (Costless Collar)

## The claim under test

- **The folklore.** "Own the index, buy a protective put ~5% out of the money, and sell a
  call whose premium exactly pays for the put — a **zero-cost (costless) collar**. You keep
  the crash protection and it doesn't cost you a dime." This is a real, widely-marketed
  structure sold by advisors as a "have your cake and eat it too" hedge, and it is a real
  published index concept, not just internet folklore.
- **The published benchmark.** The **CBOE S&P 500 95-110 Collar Index (CLL)** methodology
  (Cboe Global Markets, cboe.com/index/dashboard/CLL) formalizes exactly this idea: hold the
  S&P 500, buy a 3-month 5% OTM put, and partially finance it by selling a series of 1-month
  ~10% OTM calls. CLL is **not available on yfinance** (delisted/unlisted on our data
  provider — confirmed empirically before writing this study), so this study builds its own
  **stylized monthly analogue** rather than replaying CLL's exact roll schedule; see "What we
  measure" below for how ours differs and why that's stated up front, not hidden.
- **The academic anchor on the underlying mechanism.** Whaley (2002, *Return and risk of
  CBOE covered call index strategies*, Journal of Portfolio Management) is the foundational
  study of buy-write (covered call) risk/return; Israelov & Nielsen (2015, *Covered calls
  uncovered*, Journal of Investment Management) show a covered call's apparent risk reduction
  is mostly equity-beta reduction sold at an unattractive price, not "alpha" — the same logic
  a collar inherits on its call leg (see sibling [337-covered-call-etf](../337-covered-call-etf/)).

## What we measure, and the honesty rails

- **No live option chain — a stylized Black-Scholes model, stated loudly.** yfinance carries
  no historical SPY option prices. Every month: the put is struck 5% OTM; the call strike is
  solved (bisection on the Black-Scholes formula) so its premium exactly equals the put's —
  the model's operational definition of "costless." Both legs are priced off **trailing
  63-session realized volatility**, our proxy for the implied-vol input a real market maker
  would quote. This is an approximation, and its main failure mode is named explicitly:
- **The flat-vol approximation ignores the real skew.** SPX/SPY-linked index options carry a
  well-documented, persistent **negative volatility skew** — OTM puts trade at meaningfully
  higher implied vol than equally-OTM calls (Bollen & Whaley 2004, *Does net buying pressure
  affect the shape of implied volatility functions?*, JF; Bakshi, Kapadia & Madan 2003 on the
  risk-neutral skewness of index returns). Our flat (skew-free) model prices both legs off the
  **same** volatility number, so it likely UNDERSTATES how expensive a real 5% OTM put is
  relative to an equally-distant call — meaning a real listed collar probably needs to sell a
  **more distant** (higher-strike) call than our model implies to raise the same premium, i.e.
  a real cap is likely somewhat WIDER than our modeled ~5.9%–6.7%. We flag this rather than
  silently absorb it into the headline. Separately (and less consequentially, given our own
  vol-invariance finding below): realized vol runs on average BELOW implied vol (the variance
  risk premium; Carr & Wu 2009, *Variance risk premia*, RFS) — but because both legs share
  the same vol input in our construction, a uniform vol substitution moves the equalizing
  strike far less than one might expect.
- **A genuine model quirk, reported honestly.** The solved cap is nearly **vol-invariant** at
  this floor (5%), tenor (1 month) and rate (3%) — 5.81%–6.67% across a realized-vol range of
  5.6%–70.7%. This is an emergent property of equalizing two OTM Black-Scholes premiums at a
  short tenor with modest moneyness, not a bug; we did not smooth it away or re-parameterize
  to make it "look more sensitive." It also means the model does **not** predict tighter caps
  in calm regimes and looser caps in turbulent ones — a real skewed, term-structured market
  might behave differently, another reason this is a stylized approximation, not a chain.
- **One documented execution lag.** The volatility input (and hence the floor/cap) for month
  *t* is measured through the close of month *t−1* — known before month *t* begins, zero
  look-ahead. Costs: 2 legs (put + call) rolled monthly, one-way cost × NAV each, charged
  once per month; the equity leg is buy-and-hold and untraded (no cost).
- **The "is it just 2 lucky crashes" check.** We explicitly test the sample **excluding only**
  the two windows the claim's own marketing invokes (2008 GFC, 2020 COVID) — and deliberately
  leave the 2000–2002 dot-com bear market **in**, so the check isn't tilted to help either
  side. Newey-West (1987) HAC *t*, checked across 4 lag choices for robustness.

## Data sources

- **SPY daily raw OHLC + dividend-adjusted close** — yfinance (no key), cached under
  `_cache/` (`cc_spy.csv`), 1993-02-01 → 2026-06-30.
- **CBOE Collar Index (CLL) methodology** (context only, not backtested here — unavailable on
  yfinance): https://www.cboe.com/index/dashboard/cll/
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [617-crash-insurance-cost](../617-crash-insurance-cost/) — **buying** crash insurance
  outright (a naked long put / tail-risk fund), no financing leg. This study's put is only
  HALF the structure; the other half — selling a call to pay for it — is exactly what
  617 doesn't do, and exactly what turns "insurance" into "collar."
- [658-put-write-premium](../658-put-write-premium/) — **selling** puts for premium income
  (a cash-secured put write), the mirror-image short-vol trade. This study sells a CALL and
  buys a PUT while owning the underlying; the risk profile (capped upside, floored downside)
  is the opposite shape from a put write (capped downside via the strike, unlimited-ish
  upside above the premium).
- [337-covered-call-etf](../337-covered-call-etf/) — own the index, sell a call, **no
  protective put**. That structure has capped upside AND full downside exposure — this study
  adds the put leg specifically to remove the downside 337 still carries, which is the whole
  point of the "collar" vs. plain "buy-write" distinction, and the whole reason the "free"
  claim gets made in the first place.
- [99-safety-net](../99-safety-net/) — a **trailing stop** (a rules-based exit, no options).
  Different mechanism entirely: 99 protects by selling the asset and re-buying it later
  (timing risk, whipsaw); this study protects by owning a contractual floor while still
  holding the asset through the drawdown (no re-entry risk, but a hard cap on the upside).

None of the siblings test the **combined, "self-financing" collar structure** — the covered
call plus the protective put, priced so the premiums net to zero — which is this study's own
axis.
