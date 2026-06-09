"""Study 13 — Crimson-Hour: does a red opening hour + an IB-high rejection really call the close?

The reusable pieces, in the desk's usual split:

    * :mod:`data` — the **session panel**, one row per trading day carrying the morning
      (first-hour return, IB high/low timing) and the close (rest-of-day and full-session
      returns): a synthetic generator with a **baked-in momentum** and a deliberately
      *uninformative* IB-rejection flag (offline, deterministic, what the tests assert on), and
      a cache-only reader that reduces real Yahoo intraday bars to the same panel.
    * :mod:`signals` — the two morning tells edgeful stacks (OC-red, IB-high-rejected) and their
      confluence, plus the two outcomes a study must keep apart: the full **session** close
      (mechanically half-decided by 10:30) and the **rest-of-day** continuation (the real forecast).
    * :mod:`decompose` — the teardown: conditional close-red rates with Wilson intervals and a
      beta-binomial posterior (the honest read on "22 of 25"), the mechanical-vs-forecast split,
      the test of whether IB-rejection adds anything over OC-red, and a forking-paths Monte-Carlo
      of how selection + small samples inflate a modest edge into an 88% headline.
"""
