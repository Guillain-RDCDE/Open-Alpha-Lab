"""Abstract broker interface — keep the strategy broker-agnostic.

Any concrete broker (MT5, Alpaca, IBKR) implements these few methods. The
overnight scheduler talks only to this interface, so swapping venues never
touches strategy logic.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: float
    price: float | None = None  # None -> market order
    note: str = ""


class BrokerBase(ABC):
    """Minimal surface the overnight loop needs."""

    @abstractmethod
    def connect(self) -> None:
        """Open the session (lazy-import heavy SDKs here, not at module load)."""

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def is_market_open(self, symbol: str) -> bool:
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> tuple[float, float]:
        """Return (bid, ask)."""

    @abstractmethod
    def submit(self, order: Order) -> dict:
        """Submit an order; return a broker-specific result/confirmation dict."""

    @abstractmethod
    def position(self, symbol: str) -> float:
        """Net signed quantity currently held in ``symbol`` (0 if flat)."""

    @abstractmethod
    def overnight_financing_bps(self, symbol: str) -> float:
        """Per-night financing/swap cost in bps for a long position.

        Critical for CFD/MT5: this swap can erase the entire overnight edge, so
        the loop should refuse to trade if it is too large. Return 0 for cash
        equity venues with no overnight financing on unleveraged longs.
        """
