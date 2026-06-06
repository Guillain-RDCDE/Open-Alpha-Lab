"""MetaTrader 5 connector — TEMPLATE (dry-run by default, live untested).

This implements BrokerBase against MetaTrader5. It defaults to ``dry_run=True``
so importing and running it touches no real account. The live path requires a
running MT5 terminal + broker account and has NOT been executed end-to-end.

CFD reality check, baked into the loop: before buying, we read
``symbol_info(sym).swap_long`` and refuse if the overnight swap exceeds the
expected overnight edge. On CFDs the swap alone can turn the trade negative.
"""

from __future__ import annotations

import datetime as dt
import time

from .base import BrokerBase, Order, Side


class MT5Connector(BrokerBase):
    def __init__(self, dry_run: bool = True, deviation: int = 20, magic: int = 20260601):
        self.dry_run = dry_run
        self.deviation = deviation
        self.magic = magic
        self._mt5 = None  # imported lazily in connect()
        self._fake_positions: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            print("[MT5] dry_run=True — no terminal connection made.")
            return
        import MetaTrader5 as mt5  # lazy: only needed for live trading

        self._mt5 = mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        print("[MT5] connected:", mt5.terminal_info())

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    # -- market data -------------------------------------------------------
    def is_market_open(self, symbol: str) -> bool:
        if self.dry_run or self._mt5 is None:
            return True
        info = self._mt5.symbol_info(symbol)
        return bool(info and info.trade_mode != 0)

    def get_quote(self, symbol: str) -> tuple[float, float]:
        if self.dry_run or self._mt5 is None:
            return (100.0, 100.02)  # fake bid/ask
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick for {symbol}")
        return (tick.bid, tick.ask)

    def overnight_financing_bps(self, symbol: str) -> float:
        if self.dry_run or self._mt5 is None:
            return 0.0
        info = self._mt5.symbol_info(symbol)
        if info is None:
            return 0.0
        # swap_long is broker/symbol specific (points or %/year depending on
        # swap_mode). Treat as an indicative magnitude; verify per broker.
        return float(getattr(info, "swap_long", 0.0))

    # -- orders ------------------------------------------------------------
    def submit(self, order: Order) -> dict:
        if self.dry_run or self._mt5 is None:
            sign = 1.0 if order.side is Side.BUY else -1.0
            self._fake_positions[order.symbol] = (
                self._fake_positions.get(order.symbol, 0.0) + sign * order.quantity
            )
            result = {"dry_run": True, "order": order.__dict__}
            print(f"[MT5] (dry) {order.side.value} {order.quantity} {order.symbol} :: {order.note}")
            return result

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(order.symbol)
        price = order.price or (tick.ask if order.side is Side.BUY else tick.bid)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": float(order.quantity),
            "type": mt5.ORDER_TYPE_BUY if order.side is Side.BUY else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": order.note[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        return res._asdict() if hasattr(res, "_asdict") else {"raw": res}

    def position(self, symbol: str) -> float:
        if self.dry_run or self._mt5 is None:
            return self._fake_positions.get(symbol, 0.0)
        positions = self._mt5.positions_get(symbol=symbol)
        if not positions:
            return 0.0
        net = 0.0
        for p in positions:
            net += p.volume if p.type == self._mt5.POSITION_TYPE_BUY else -p.volume
        return net


def run_overnight_loop(
    broker: BrokerBase,
    symbol: str,
    quantity: float,
    close_time: dt.time = dt.time(15, 55),
    open_time: dt.time = dt.time(9, 35),
    max_swap_bps: float = 5.0,
    poll_seconds: int = 30,
    iterations: int | None = None,
):
    """Buy ~5 min before the close, sell ~5 min after the open. Loop.

    Safety rails:
      * refuses to open if ``broker.overnight_financing_bps`` exceeds
        ``max_swap_bps`` (the CFD swap trap);
      * only ever holds one overnight unit, flat during the day.

    ``iterations`` bounds the loop for testing (None = run forever). This is a
    TEMPLATE: validate every assumption on a demo account before real capital.
    """
    broker.connect()
    count = 0
    try:
        while iterations is None or count < iterations:
            now = _now_time()
            held = broker.position(symbol) != 0.0

            if not held and _within(now, close_time, poll_seconds):
                swap = broker.overnight_financing_bps(symbol)
                if abs(swap) > max_swap_bps:
                    print(f"[loop] swap {swap} bps > max {max_swap_bps}; SKIP overnight.")
                else:
                    broker.submit(Order(symbol, Side.BUY, quantity, note="overnight entry @close"))

            elif held and _within(now, open_time, poll_seconds):
                broker.submit(Order(symbol, Side.SELL, quantity, note="overnight exit @open"))
                count += 1

            time.sleep(poll_seconds)
    finally:
        broker.disconnect()
    return count


def _now_time() -> dt.time:  # pragma: no cover - wall-clock
    return dt.datetime.now().time()


def _within(now: dt.time, target: dt.time, tol_seconds: int) -> bool:
    """True if ``now`` is within ``tol_seconds`` after ``target``."""
    today = dt.date(2000, 1, 1)
    d_now = dt.datetime.combine(today, now)
    d_tgt = dt.datetime.combine(today, target)
    return 0 <= (d_now - d_tgt).total_seconds() <= tol_seconds
