"""Broker connectors. The interface (BrokerBase) is deliberately swappable so
the same overnight loop can run on MT5 today and Alpaca/IBKR later without
touching the strategy code."""

from .base import BrokerBase, Order, Side

__all__ = ["BrokerBase", "Order", "Side"]
