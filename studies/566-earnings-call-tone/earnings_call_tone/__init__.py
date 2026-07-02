"""Study 566 — Earnings-Call-Tone (a linguistic cousin of PEAD).

The claim: the *net emotional tone* of a company's earnings-call transcript — how upbeat vs
guarded management sounds, measured as (positive − negative) sentiment-word share — predicts the
stock's **post-call drift** (the cumulative abnormal return over the days *after* the call). It is
the NLP relative of post-earnings-announcement drift: instead of the *numeric* surprise driving a
slow drift (Study 363), the *tone* of the words does.

This study is **synthetic-only by design.** There is no free, no-key retail feed of scored
earnings-call transcripts joined to event-time abnormal returns (that lives in paid NLP vendors —
RavenPack, AlphaSense, S&P's Textline — and hand-scored academic samples). So the reproducible core
is a deterministic, seeded tone panel with a single knob (``tone_beta``) that plants the tone→drift
link; the strategy engine catches the planted effect and reads flat at the null. Because a `REAL`
Signal requires a robust *t* ≥ 2 on a **real** tape, and we have none, this study is **capped at
`WEAK`** — the literature (Loughran-McDonald 2011; Price et al. 2012; Mayew-Venkatachala 2012)
supports the effect, but this desk's own tape cannot certify it. The data-availability limitation is
stated openly on the SIGNAL axis, exactly like the desk's lego-returns / whisky-cask studies.

Distinct from the desk's numeric-surprise drifts — [Study 363 PEAD-drift](../363-pead-drift/) and
[Study 534 revenue-surprise-drift](../534-revenue-surprise-drift/) drift on the *number*; this
study drifts on the *words*. Distinct too from the aggregate-mood studies —
[Study 259 news-tone](../259-news-tone/) and [Study 392 glassdoor-sentiment](../392-glassdoor-sentiment/)
score macro/employer mood, not a firm's own earnings-call transcript against its own post-call CAR.
"""
