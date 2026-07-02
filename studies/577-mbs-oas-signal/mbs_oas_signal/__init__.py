"""Study 577 — MBS-OAS-Signal (mortgage-spread widening as a risk-off leading indicator).

The claim (a cross-asset folklore staple): the **option-adjusted spread** on agency
mortgage-backed securities — the extra yield MBS pay over Treasuries after stripping out the
embedded prepayment option — is a *leading* indicator of stress. When MBS OAS **widens**, the
story goes, mortgage investors are demanding more compensation for risk, and that risk-off signal
front-runs weakness in equities and corporate credit. So a rising OAS should predict *low* forward
equity/credit returns, and a strategy that de-risks when OAS widens should dodge drawdowns.

This desk cannot fetch a clean, long, free daily MBS OAS series (the canonical series are
vendor/index products — ICE BofA, Bloomberg — behind licences; FRED's mortgage series are yields,
not option-adjusted spreads). So this study is **synthetic-only**: a deterministic generator plants
(or withholds) an OAS→forward-return lead, and the engine measures whether it can recover it. That
data-availability wall is stated openly on the SIGNAL axis, and it caps the study below `REAL`
(which needs a robust t >= 2 on a genuine tape).

Distinct from the desk's spread neighbours: [Study 115 — Credit-Spreads](../115-credit-spreads/)
tests the *corporate* HY/IG spread as a state variable; this study is specifically the
**agency-MBS option-adjusted spread** as a cross-asset *lead*. [Study 05 — Twin-Spread](../05-twin-spread/)
is a relative-value pair trade, not a macro risk-off signal.
"""
