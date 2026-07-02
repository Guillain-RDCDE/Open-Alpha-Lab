"""Study 565 — Filing-Readability (Loughran-McDonald text anomaly).

The *readability anomaly*: firms whose 10-K annual reports are **harder to read** — a higher fog
index, a longer document, a fatter file size — go on to earn **lower** future returns and carry
**higher** risk (Loughran & McDonald 2014, *Measuring Readability in Financial Disclosures*, JF).
The story is obfuscation: managers with bad news to bury write longer, murkier filings, and the
market underreacts to the buried signal, so the murky-filing firms drift down.

We build a synthetic cross-sectional **readability panel** (fog / length / file-size legs) versus
forward returns, calibrated to the LM literature (a single knob plants the effect), run the
information-coefficient test + a label-shuffle placebo + a seed-robust synthetic positive control,
and expose a **real-tape stub** that is empty by construction — the free retail stack cannot reach
a clean point-in-time 10-K text panel (EDGAR full-text + a survivorship-free CRSP-style return
tape), so this study is **synthetic-only** and can never earn `REAL` (that needs a robust t >= 2 on
a real tape). The data-availability limitation is stated openly on the SIGNAL axis.

Distinct from this desk's sentiment studies ([257 AAII-sentiment](../257-aaii-sentiment/),
[335 Buzz-sentiment-ETF](../335-buzz-sentiment-etf/), [392 Glassdoor-sentiment](../392-glassdoor-sentiment/)):
those test *tone / opinion* signals; Study 565 is the LM **readability / length** anomaly — a
structural property of the filing (how hard it is to read), not its sentiment.
"""
