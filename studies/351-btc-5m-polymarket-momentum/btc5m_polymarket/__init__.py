"""Study 351 — BTC 5-minute Polymarket momentum (the "$300 -> $14,000" bot).

Tears down a viral GitHub bot that bets the favoured side of Polymarket BTC Up/Down
5-minute markets into the close. The momentum signal is real (continuation >> a coin); the
edge is a mirage (the favourite is priced at ~fair, 0.97-0.99, so ``w - p <= 0`` net of the
spread); and the "$300 -> $14,000" is a martingale survivor story.

See :mod:`btc5m_polymarket.data` (real 1m BTC tape + synthetic control + live CLOB capture)
and :mod:`btc5m_polymarket.strategy` (continuation win-rate, EV, bankroll Monte-Carlo)."""

from . import data, strategy

__all__ = ["data", "strategy"]
