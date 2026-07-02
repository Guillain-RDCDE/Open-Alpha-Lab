"""Study 567 — Uncertainty-Word-Count (the LM uncertainty dictionary as a vol/return signal).

The claim: the *density of uncertainty / hedging words* in a firm's disclosure — "uncertain",
"maybe", "could", "risk", "approximately", "depend" — measured against the Loughran-McDonald (2011)
**Uncertainty** word list, predicts **higher future volatility** and **lower future returns**. When
management piles on hedges, the market becomes less sure about the firm, and that ambiguity shows up
as realised-vol spikes (and, the softer half of the claim, a modest return drag).

This study is **synthetic-only by design.** There is no free, no-key retail feed of parsed 10-K/10-Q
filings (or call transcripts) scored against the LM uncertainty dictionary and joined to *forward*
realised volatility. That product lives in paid NLP vendors and hand-parsed academic samples
(EDGAR full-text + a licensed lexicon pipeline). So the reproducible core is a deterministic, seeded
firm-event panel with a single knob (``uncert_beta``) that plants the uncertainty-word → forward-vol
link; the engine catches the planted effect and reads flat at the null. Because a `REAL` Signal
requires a robust *t* ≥ 2 on a **real** tape, and we have none, this study is **capped at `WEAK`** —
the literature (Loughran-McDonald 2011; Campbell et al. 2014; Kravet-Muslu 2013; Bochkay et al. 2020)
supports the effect, but this desk's own tape cannot certify it. The data-availability limitation is
stated openly on the SIGNAL axis, exactly like the desk's lego-returns / whisky-cask / earnings-call
-tone studies.

The honesty pivot is **confounding by recent volatility.** Firms that are *already* jumpy tend to
write hedgier disclosures (uncertain firms sound uncertain), so a naive uncertainty→forward-vol
regression double-counts the *persistence of volatility* (vol is autocorrelated). The engine
therefore controls for trailing realised vol and reports the *residual* uncertainty-word slope — the
part of the hedging density orthogonal to how volatile the firm already was — as the honest headline.

Distinct from the desk's tone / sentiment studies — [Study 566 earnings-call-tone](../566-earnings-call-tone/)
scores *net emotional tone* (positive − negative) against post-call **drift** (a return signal);
[Study 259 news-tone](../259-news-tone/) and [Study 392 glassdoor-sentiment](../392-glassdoor-sentiment/)
score aggregate mood. This study scores a *single, orthogonal* lexicon — the LM **Uncertainty**
list (hedging/ambiguity words, not positive/negative sentiment) — against **forward realised
volatility** (a risk signal), with returns as the secondary axis. The outcome is *vol*, not tone.
"""
