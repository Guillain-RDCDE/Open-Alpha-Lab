# References & literature map — Study 784 (Analyst-Cluster)

## The claim under test

- **The folklore.** "When *everybody* upgrades a stock in the same week, fade it — the
  sell-side pile-in is the last marginal buyer, so a cluster of same-week price-target
  hikes marks a local sentiment top and the name underperforms afterwards." A perennial
  contrarian trade: analyst herding is treated as a crowdedness / capitulation signal.
- **The bullish counter-story.** The opposite reading is that a cluster of upgrades is a
  *real* information event and the stock keeps drifting up — sell-side momentum, or plain
  post-earnings-announcement drift on the print that triggered the upgrades. This study
  lets NVDA's tape adjudicate between "fade it" and "ride it."
- **LABELLED PROXY.** There is no free, survivorship-clean, point-in-time analyst-upgrade
  feed in this repo, so the cluster week is **proxied by NVDA's real quarterly earnings
  week** — the canonical trigger of a same-week price-target-hike wave (financial media
  reliably runs "N analysts raised targets on NVIDIA" after each print, routinely 20-40
  firms in the AI-boom years). The earnings dates are real, publicly-verifiable facts and
  are known ~3-4 weeks ahead, so the entry is calendar-known and zero-look-ahead. The
  honest confound: this proxy inherits **post-earnings-announcement drift**, which we
  report openly rather than hide (see [`data.py`](../analyst_cluster/data.py)).

## What the literature actually says

- **Analyst-recommendation profitability & herding** — Womack (1996, *JF*) on the price
  impact of recommendation changes; Barber, Lehavy, McNichols & Trueman (2001, *JF*),
  "Can investors profit from the prophets?" — buying up-rated / selling down-rated names
  earns abnormal returns *gross* but the edge is eaten by turnover and costs. Jegadeesh &
  Kim (2010, *RFS*) document analyst **herding** — analysts cluster their revisions — which
  is exactly the "same-week cluster" this study proxies.
- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard &
  Thomas (1989, 1990, *JAR / JAE*). The canonical "prices drift in the direction of the
  surprise *after* a scheduled print." Because our cluster proxy is the earnings week, a
  measured "fade" and a measured "drift" live on the same window — PEAD is the null we
  cannot fully separate on a single name.
- **Recommendation-change drift** — Stickel (1995, *FAJ*); Gleason & Lee (2003, *AR*) find
  prices under-react to and then drift *with* analyst revisions — evidence *against* a fade
  and *for* continuation, consistent with what NVDA shows here.
- **Attention & crowding** — Barber & Odean (2008, *RFS*) on attention-driven buying; Da,
  Engelberg & Gao (2011, *JF*) on search attention. These motivate *why* a cluster might be
  a crowding top — but attention is not, by itself, a tradable fade.
- **Single-name selection caveat** — this is one hand-picked mega-winner (NVDA rose ~300×
  over 2016-2025). Any "abnormal vs SPY" window on such a name is positive by selection
  (see the placebo: even *random* 2-week NVDA windows beat SPY by ~+2%). The finding here
  is not evidence for a generalizable, cross-sectional strategy.

## Data & method

- **Real tape:** `NVDA` and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. NVDA's very high
  beta is why we measure the *abnormal* return `NVDA − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across 39 independent,
  non-overlapping cluster events (the correct unit — not a daily panel); Wilson hit-rate
  interval; a 20-seed × 200-draw random-window placebo per cut; a leave-one-out jackknife;
  a costed net leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  pre-cluster run-up (and optional post-cluster fade) — the detector must recover a planted
  bump and stay quiet on the null. See [`strategy.py`](../analyst_cluster/strategy.py).

*Womack, K. (1996). **JF**. · Barber, Lehavy, McNichols & Trueman (2001). **JF**. ·
Jegadeesh, N. & Kim, W. (2010). **RFS**. · Ball, R. & Brown, P. (1968). **JAR**. · Bernard,
V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Stickel, S. (1995). **FAJ**. · Gleason, C. &
Lee, C. (2003). **The Accounting Review**. · Barber, B. & Odean, T. (2008). **RFS**. · Da,
Z., Engelberg, J. & Gao, P. (2011). **JF**.*
