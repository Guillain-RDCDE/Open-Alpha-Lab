"""Study 377 — Bid-Ask-Bounce (is short-horizon "mean reversion" real edge, or the bounce?).

Roll (1984) showed that the bid-ask bounce alone — transaction prices flipping between bid and
ask around an efficient random-walk price — injects a *negative* lag-1 autocovariance into
observed returns, ``cov1 = -(spread/2)**2``. That looks exactly like one-day "mean reversion,"
but it carries **no tradable edge**: a reversal book that tries to harvest it has to cross the
very spread that creates it. So the famous short-horizon reversion is, in large part, the bounce
in disguise.

The offline CORE is the Roll model (:mod:`bid_ask_bounce.data.synthetic_roll`): a true price that
is a pure random walk (zero reversion) plus a bid-ask bounce reproduces the illusion, and a
planted *genuine* AR(1) reversion is the positive control — only it survives the spread net of
cost. The real-tape PROXY (:mod:`bid_ask_bounce.data.load_real`) runs the same reversal book on a
small-cap basket (where spreads, hence the bounce, are widest) and measures gross-vs-net.

See :mod:`bid_ask_bounce.strategy` for the Roll spread estimator, the reversal book, and the
inference (Welch t, block-bootstrap null, win-rate, gross/net of one-way half-spread costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
