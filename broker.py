"""
Broker adapter interface and implementations.

Provides a uniform API for order execution across platforms.
Plug in whichever broker your prop firm uses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("bot.broker")


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    quantity: int          # Number of contracts
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tag: str = ""          # Custom label for tracking


@dataclass
class OrderFill:
    order_id: str
    symbol: str
    side: Side
    quantity: int
    fill_price: float
    status: OrderStatus
    timestamp: float = field(default_factory=time.time)
    tag: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int          # Positive = long, negative = short
    avg_entry: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class AccountState:
    equity: float
    balance: float         # Cash balance (no open PnL)
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float       # PnL since session open
    positions: list[Position] = field(default_factory=list)


@dataclass
class Bar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: int


class BrokerAdapter(ABC):
    """Abstract broker interface. Implement for your platform."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean disconnect."""

    @abstractmethod
    async def get_account(self) -> AccountState:
        """Fetch current account state."""

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderFill:
        """Submit order and return fill."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""

    @abstractmethod
    async def cancel_all(self, symbol: str) -> int:
        """Cancel all open orders for symbol. Returns count cancelled."""

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""

    @abstractmethod
    async def flatten(self, symbol: str) -> Optional[OrderFill]:
        """Close all positions for symbol at market."""

    @abstractmethod
    async def get_bars(self, symbol: str, interval_minutes: int,
                        count: int) -> list[Bar]:
        """Fetch recent OHLCV bars."""

    @abstractmethod
    async def stream_bars(self, symbol: str, interval_minutes: int,
                           callback) -> None:
        """Stream live bars. callback(bar: Bar) called on each new bar."""

    @abstractmethod
    async def get_tick(self, symbol: str) -> float:
        """Get current market price (last trade or mid)."""


# ── Paper broker for testing ─────────────────────────────────────────

class PaperBroker(BrokerAdapter):
    """
    Simulated broker for backtesting and dry-run mode.
    Fills at current price with no slippage.
    """

    def __init__(self, initial_equity: float = 100_000,
                 tick_value: float = 20.0, tick_size: float = 0.25):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.balance = initial_equity
        self.daily_pnl = 0.0
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}
        self.tick_value = tick_value
        self.tick_size = tick_size
        self._order_counter = 0
        self._current_prices: dict[str, float] = {}
        self._bar_callbacks: dict[str, list] = {}
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        log.info("Paper broker connected (equity=$%.0f)", self.equity)

    async def disconnect(self) -> None:
        self._connected = False
        log.info("Paper broker disconnected")

    async def get_account(self) -> AccountState:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return AccountState(
            equity=self.balance + unrealized,
            balance=self.balance,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            daily_pnl=self.daily_pnl,
            positions=list(self.positions.values()),
        )

    async def submit_order(self, order: OrderRequest) -> OrderFill:
        self._order_counter += 1
        price = self._current_prices.get(order.symbol, 0)

        if price == 0:
            return OrderFill(
                order_id=f"paper-{self._order_counter}",
                symbol=order.symbol, side=order.side,
                quantity=order.quantity, fill_price=0,
                status=OrderStatus.REJECTED, tag=order.tag)

        # Process fill
        pos = self.positions.get(order.symbol)
        multiplier = self.tick_value / self.tick_size  # Point value

        if pos is None or pos.quantity == 0:
            # New position
            signed_qty = order.quantity if order.side == Side.BUY else -order.quantity
            self.positions[order.symbol] = Position(
                symbol=order.symbol, quantity=signed_qty,
                avg_entry=price, unrealized_pnl=0, realized_pnl=0)
        else:
            new_side = 1 if order.side == Side.BUY else -1
            new_signed = new_side * order.quantity

            if (pos.quantity > 0 and new_side < 0) or \
               (pos.quantity < 0 and new_side > 0):
                # Closing/reducing
                close_qty = min(abs(pos.quantity), order.quantity)
                pnl = close_qty * (price - pos.avg_entry) * (
                    1 if pos.quantity > 0 else -1) * multiplier
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                self.balance += pnl
                remaining = abs(pos.quantity) - close_qty
                if remaining == 0:
                    del self.positions[order.symbol]
                else:
                    pos.quantity = remaining * (1 if pos.quantity > 0 else -1)
                    pos.realized_pnl += pnl
            else:
                # Adding to position
                total_cost = pos.avg_entry * abs(pos.quantity) + price * order.quantity
                pos.quantity += new_signed
                pos.avg_entry = total_cost / abs(pos.quantity)

        fill = OrderFill(
            order_id=f"paper-{self._order_counter}",
            symbol=order.symbol, side=order.side,
            quantity=order.quantity, fill_price=price,
            status=OrderStatus.FILLED, tag=order.tag)

        log.info("FILL  %s %d %s @ %.2f  [%s]",
                 order.side.value, order.quantity, order.symbol,
                 price, order.tag)
        return fill

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def cancel_all(self, symbol: str) -> int:
        return 0

    async def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    async def flatten(self, symbol: str) -> Optional[OrderFill]:
        pos = self.positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return None
        side = Side.SELL if pos.quantity > 0 else Side.BUY
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=side, quantity=abs(pos.quantity),
            tag="flatten"))

    async def get_bars(self, symbol: str, interval_minutes: int,
                        count: int) -> list[Bar]:
        return []

    async def stream_bars(self, symbol: str, interval_minutes: int,
                           callback) -> None:
        pass

    async def get_tick(self, symbol: str) -> float:
        return self._current_prices.get(symbol, 0)

    def set_price(self, symbol: str, price: float):
        """Set simulated price (call from test harness)."""
        self._current_prices[symbol] = price
        # Update unrealized PnL
        pos = self.positions.get(symbol)
        if pos and pos.quantity != 0:
            multiplier = self.tick_value / self.tick_size
            pos.unrealized_pnl = pos.quantity * (
                price - pos.avg_entry) * multiplier

    def reset_daily(self):
        """Call at session open to reset daily PnL."""
        self.daily_pnl = 0.0
