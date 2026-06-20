"""Study 345 — Survivorship-Bias.

A research-method demo: take ONE cross-sectional strategy and run it twice — once on a
panel that includes the companies that *died* (delisted on bankruptcy / collapse), and
once on the survivors-only panel you get if you build a universe from *today's*
membership and project it backwards. The gap between the two reads is the bias.

- ``data``     — a deterministic offline panel generator that plants delisting
                 (firms whose price drifts to ~0 and then leaves the tape), plus a
                 cache-first ``load_real`` that runs the same demo on a real tape
                 through the desk's opt-in survivorship guard.
- ``strategy`` — the engine: a simple cross-sectional book, honest HAC-t and block-
                 bootstrap inference, one-way costs × NAV, one documented execution lag,
                 and the survivor-vs-full delta that *is* the bias.
"""
