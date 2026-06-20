# References & literature map — Study 319 (Lockup-Expiry)

The IPO lock-up expiry is one of the most-studied "predictable" corporate-calendar events.
The early literature found a small, real abnormal sag; the modern read is that the effect
shrank as the unlock became fully anticipated and arbitraged.

## The foundational lock-up studies

- **Field, L. C., & Hanka, G. (2001).** *"The Expiration of IPO Share Lockups."* Journal of
  Finance 56(2), 471–500. The canonical paper: a statistically significant **abnormal return
  of roughly −1.5%** in the three days around lock-up expiry, larger (≈ −2.5%) for
  venture-backed firms, accompanied by a permanent **40% jump in trading volume**. Our
  market-adjusted event study is the same design, on a recent basket.

- **Bradley, D. J., Jordan, B. D., Roten, I. C., & Yi, H.-C. (2001).** *"Venture Capital and
  IPO Lockup Expiration: An Empirical Analysis."* Journal of Financial Research 24(4),
  465–493. Confirms a negative expiry-window return concentrated in VC-backed IPOs, and links
  the sag's size to the proportion of shares unlocking.

- **Brav, A., & Gompers, P. A. (2003).** *"The Role of Lockups in Initial Public Offerings."*
  Review of Financial Studies 16(1), 1–29. Studies *why* lock-ups exist (commitment /
  signalling / moral-hazard mitigation) and documents the price reaction at expiry; argues the
  effect is hard to reconcile with semi-strong efficiency since the date is known in advance.

## Anticipation, decay, and the "predictability puzzle"

- **Ofek, E., & Richardson, M. (2000).** *"The IPO Lock-Up Period: Implications for Market
  Efficiency and Downward Sloping Demand Curves."* NYU working paper. Frames the sag as
  evidence for downward-sloping demand curves for stock (a supply shock moves price), the
  mechanism the folklore short is implicitly betting on.

- **Cao, C., Field, L. C., & Hanka, G. (2004).** *"Does Insider Trading Impair Market
  Liquidity? Evidence from IPO Lockup Expirations."* Journal of Financial and Quantitative
  Analysis 39(1), 25–46. Documents the liquidity and microstructure changes at expiry — the
  spread/borrow context that makes the event hard to trade net of costs.

## Context: IPO returns the rest of this desk's IPO work covers

- **Ritter, J. R. (1991).** *"The Long-Run Performance of Initial Public Offerings."* Journal
  of Finance 46(1), 3–27. The multi-year IPO underperformance result — distinct from the
  single-event lock-up sag, and the subject of **Study 219 (IPO-Pop)** on this desk, which
  also measures the first-day pop. Study 219 explicitly flagged lock-up expiry as untested
  *"going further"* — this study is that follow-up.

## Method

- **MacKinlay, A. C. (1997).** *"Event Studies in Economics and Finance."* Journal of Economic
  Literature 35(1), 13–39. The standard reference for the abnormal-return / AAR / CAR event-
  study machinery and its cross-sectional test statistics that this study implements.

- **Politis, D. N., & Romano, J. P. (1994).** *"The Stationary Bootstrap."* JASA 89(428),
  1303–1313. Basis for the block-bootstrap CIs used on the cross-section of per-event CARs.

- **Newey, W. K., & West, K. D. (1987).** *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."* Econometrica 55(3),
  703–708. The HAC standard error behind every *t*-stat reported here.

## Where the folklore lives

Trading blogs and "IPO calendar" services routinely market shorting the lock-up expiry as an
edge ("180 days after the IPO, insiders dump — short it"). The early academic evidence is real
but small (≈ −1.5%) and the date is public; this study tests whether any harvestable sag
survives on a recent (2019–2024) basket once the event is fully anticipated and costed.
