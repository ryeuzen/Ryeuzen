"""
TopstepX (ProjectX) broker adapter.

Connects to TopstepX via the tsxapi4py library for:
  - Authentication (API key + username)
  - Order placement (market, limit, stop)
  - Real-time market data streaming (SignalR WebSocket)
  - Position management
  - Historical bars

Requires:
  pip install git+https://github.com/mceesincus/tsxapi4py.git

  .env file with:
    TRADING_ENVIRONMENT=DEMO  (or LIVE)
    API_KEY=your_topstep_api_key
    USERNAME=your_topstep_username
    CONTRACT_ID=CON.F.US.MNQ.M25  (current front month)
    ACCOUNT_ID_TO_WATCH=12345
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import partial
from typing import Optional

from broker import (
    BrokerAdapter, AccountState, Bar, OrderFill, OrderRequest,
    OrderStatus, OrderType, Position, Side,
)

log = logging.getLogger("bot.topstep")

# Contract ID mapping for TopstepX
# Format: CON.F.US.{ROOT}.{MONTH_CODE}{YY}
# Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
#              N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
TOPSTEP_SYMBOLS = {
    "NQ": "NQ",
    "MNQ": "MNQ",
    "ES": "ES",
    "MES": "MES",
}


class TopstepBroker(BrokerAdapter):
    """
    TopstepX broker adapter using tsxapi4py.

    Wraps the ProjectX REST + SignalR WebSocket API.
    """

    def __init__(self, contract_id: str = None, account_id: int = None):
        """
        Args:
            contract_id: TopstepX contract ID (e.g. "CON.F.US.MNQ.M25")
                         If None, reads from CONTRACT_ID env var.
            account_id: TopstepX account ID. If None, reads from
                        ACCOUNT_ID_TO_WATCH env var or auto-detects.
        """
        self._contract_id = contract_id or os.getenv("CONTRACT_ID", "")
        self._account_id = account_id or int(
            os.getenv("ACCOUNT_ID_TO_WATCH", "0"))
        self._api_client = None
        self._order_placer = None
        self._data_stream = None
        self._user_stream = None
        self._data_manager = None
        self._connected = False

        # Callbacks for streaming
        self._bar_callback = None
        self._latest_quote: dict = {}
        self._latest_price: float = 0.0

        # Order tracking
        self._pending_orders: dict[str, asyncio.Future] = {}
        self._order_updates: dict[str, dict] = {}

    async def connect(self) -> None:
        """Authenticate and establish connections."""
        try:
            from tsxapipy import authenticate, APIClient
        except ImportError:
            raise ImportError(
                "tsxapi4py not installed. Run:\n"
                "  pip install git+https://github.com/mceesincus/"
                "tsxapi4py.git"
            )

        log.info("Authenticating with TopstepX...")
        token_str, acquired_at = await asyncio.to_thread(authenticate)
        self._api_client = APIClient(
            initial_token=token_str,
            token_acquired_at=acquired_at,
        )
        log.info("Authenticated successfully")

        # Auto-detect account if not set
        if self._account_id == 0:
            accounts = await asyncio.to_thread(
                self._api_client.get_accounts, only_active=True)
            if accounts:
                self._account_id = accounts[0].id
                log.info("Auto-detected account ID: %d", self._account_id)
            else:
                raise RuntimeError("No active accounts found")

        # Auto-detect contract if not set
        if not self._contract_id:
            raise RuntimeError(
                "CONTRACT_ID not set. Set it in .env or pass to constructor.\n"
                "Example: CON.F.US.MNQ.M25")

        # Set up order placer
        from tsxapipy.trading import OrderPlacer
        self._order_placer = OrderPlacer(
            api_client=self._api_client,
            account_id=self._account_id,
        )

        self._connected = True
        log.info("TopstepX connected  account=%d  contract=%s",
                 self._account_id, self._contract_id)

    async def disconnect(self) -> None:
        if self._data_stream:
            self._data_stream.stop()
        if self._user_stream:
            self._user_stream.stop()
        if self._data_manager:
            self._data_manager.stop_streaming()
        self._connected = False
        log.info("TopstepX disconnected")

    async def get_account(self) -> AccountState:
        accounts = await asyncio.to_thread(
            self._api_client.get_accounts, only_active=True)

        for acc in accounts:
            if acc.id == self._account_id:
                # Get positions
                positions = await self._get_positions()
                unrealized = sum(p.unrealized_pnl for p in positions)

                return AccountState(
                    equity=getattr(acc, 'balance', 0) + unrealized,
                    balance=getattr(acc, 'balance', 0),
                    unrealized_pnl=unrealized,
                    realized_pnl=0,  # Tracked by challenge_tracker
                    daily_pnl=0,     # Tracked by challenge_tracker
                    positions=positions,
                )

        raise RuntimeError(f"Account {self._account_id} not found")

    async def submit_order(self, order: OrderRequest) -> OrderFill:
        side_str = "Buy" if order.side == Side.BUY else "Sell"

        log.info("Submitting %s %s %d %s",
                 order.order_type.value, side_str,
                 order.quantity, self._contract_id)

        try:
            if order.order_type == OrderType.MARKET:
                order_id = await asyncio.to_thread(
                    self._order_placer.place_market_order,
                    side=side_str,
                    size=order.quantity,
                    contract_id=self._contract_id,
                )
            elif order.order_type == OrderType.LIMIT:
                order_id = await asyncio.to_thread(
                    self._order_placer.place_limit_order,
                    side=side_str,
                    size=order.quantity,
                    contract_id=self._contract_id,
                    price=order.limit_price,
                )
            elif order.order_type == OrderType.STOP:
                order_id = await asyncio.to_thread(
                    self._order_placer.place_stop_order,
                    side=side_str,
                    size=order.quantity,
                    contract_id=self._contract_id,
                    stop_price=order.stop_price,
                )
            else:
                return OrderFill(
                    order_id="", symbol=order.symbol, side=order.side,
                    quantity=order.quantity, fill_price=0,
                    status=OrderStatus.REJECTED,
                    tag=f"unsupported order type: {order.order_type}")

            if order_id:
                # For market orders, assume immediate fill at current price
                fill_price = self._latest_price or await self.get_tick(
                    order.symbol)

                log.info("Order accepted  id=%s  fill≈%.2f",
                         order_id, fill_price)

                return OrderFill(
                    order_id=str(order_id),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=fill_price,
                    status=OrderStatus.FILLED,
                    tag=order.tag,
                )
            else:
                return OrderFill(
                    order_id="", symbol=order.symbol, side=order.side,
                    quantity=order.quantity, fill_price=0,
                    status=OrderStatus.REJECTED, tag=order.tag)

        except Exception as e:
            log.error("Order failed: %s", e)
            return OrderFill(
                order_id="", symbol=order.symbol, side=order.side,
                quantity=order.quantity, fill_price=0,
                status=OrderStatus.REJECTED, tag=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await asyncio.to_thread(
                self._api_client.cancel_order, order_id=int(order_id))
            return True
        except Exception as e:
            log.error("Cancel failed: %s", e)
            return False

    async def cancel_all(self, symbol: str) -> int:
        try:
            orders = await asyncio.to_thread(
                self._api_client.search_orders,
                account_id=self._account_id)
            cancelled = 0
            for order in orders:
                if await self.cancel_order(str(order.id)):
                    cancelled += 1
            return cancelled
        except Exception as e:
            log.error("Cancel all failed: %s", e)
            return 0

    async def get_position(self, symbol: str) -> Optional[Position]:
        positions = await self._get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def flatten(self, symbol: str) -> Optional[OrderFill]:
        pos = await self.get_position(symbol)
        if pos is None or pos.quantity == 0:
            return None

        side = Side.SELL if pos.quantity > 0 else Side.BUY
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=side, quantity=abs(pos.quantity),
            tag="flatten"))

    async def get_bars(self, symbol: str, interval_minutes: int,
                        count: int) -> list[Bar]:
        """Fetch historical bars via tsxapi4py HistoricalDataUpdater."""
        try:
            # Use API client directly for bar retrieval
            # api_bar_unit: 1=Tick, 2=DailyBar, 3=MinuteBar, 4=Volume, etc.
            bars_response = await asyncio.to_thread(
                self._api_client.get_bars,
                contract_id=self._contract_id,
                bar_unit=3,  # MinuteBar
                bar_unit_number=interval_minutes,
                count=count,
            )

            bars = []
            if bars_response:
                for b in bars_response:
                    bars.append(Bar(
                        timestamp=getattr(b, 'timestamp',
                                          getattr(b, 't', time.time())),
                        open=getattr(b, 'open', getattr(b, 'o', 0)),
                        high=getattr(b, 'high', getattr(b, 'h', 0)),
                        low=getattr(b, 'low', getattr(b, 'l', 0)),
                        close=getattr(b, 'close', getattr(b, 'c', 0)),
                        volume=getattr(b, 'volume', getattr(b, 'v', 0)),
                    ))
            return bars

        except Exception as e:
            log.error("Failed to fetch bars: %s", e)
            return []

    async def stream_bars(self, symbol: str, interval_minutes: int,
                           callback) -> None:
        """
        Stream real-time bars using DataManager's candle aggregation.

        DataManager aggregates raw trades into candles and calls our
        callback with each completed bar.
        """
        from tsxapipy.pipeline import DataManager

        self._data_manager = DataManager()

        loop = asyncio.get_event_loop()

        def on_candle(candle_series):
            """Called by DataManager when a candle completes."""
            try:
                bar = Bar(
                    timestamp=time.time(),
                    open=float(candle_series.get('open', 0)),
                    high=float(candle_series.get('high', 0)),
                    low=float(candle_series.get('low', 0)),
                    close=float(candle_series.get('close', 0)),
                    volume=int(candle_series.get('volume', 0)),
                )
                # Schedule the async callback on the event loop
                if asyncio.iscoroutinefunction(callback):
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(callback(bar)))
                else:
                    loop.call_soon_threadsafe(callback, bar)
            except Exception as e:
                log.error("Candle callback error: %s", e)

        try:
            if self._data_manager.initialize_components(
                    contract_id=self._contract_id):

                # Load some history first
                self._data_manager.load_initial_history(
                    num_candles_to_load=100)

                if self._data_manager.start_streaming():
                    log.info("DataManager streaming started for %s",
                             self._contract_id)

                    # Keep streaming until session ends
                    # The bot's _run_session controls lifetime
                    while self._connected:
                        # Get latest candle data periodically
                        df = self._data_manager.get_chart_data(
                            timeframe_seconds=interval_minutes * 60)
                        if df is not None and not df.empty:
                            latest = df.iloc[-1]
                            bar = Bar(
                                timestamp=time.time(),
                                open=float(latest.get('open', 0)),
                                high=float(latest.get('high', 0)),
                                low=float(latest.get('low', 0)),
                                close=float(latest.get('close', 0)),
                                volume=int(latest.get('volume', 0)),
                            )
                            if asyncio.iscoroutinefunction(callback):
                                await callback(bar)
                            else:
                                callback(bar)

                        # Also update latest price for get_tick
                        self._update_latest_price()

                        # Refresh token periodically
                        self._data_manager.update_stream_token_if_needed()

                        await asyncio.sleep(interval_minutes * 60)

        except Exception as e:
            log.error("Streaming error: %s", e)
        finally:
            if self._data_manager:
                self._data_manager.stop_streaming()

    async def get_tick(self, symbol: str) -> float:
        """Get current price from quote stream or REST fallback."""
        if self._latest_price > 0:
            return self._latest_price

        # Fallback: start a quick quote stream
        await self._start_quote_stream()
        # Wait briefly for first quote
        for _ in range(20):
            if self._latest_price > 0:
                return self._latest_price
            await asyncio.sleep(0.25)

        return 0.0

    async def _start_quote_stream(self):
        """Start the market data WebSocket for real-time quotes."""
        if self._data_stream is not None:
            return

        try:
            from tsxapipy import DataStream

            def on_quote(quote_data: dict):
                self._latest_quote = quote_data
                # Extract price from quote
                bid = quote_data.get('bid', quote_data.get('Bid', 0))
                ask = quote_data.get('ask', quote_data.get('Ask', 0))
                last = quote_data.get('last', quote_data.get('Last', 0))
                if last:
                    self._latest_price = float(last)
                elif bid and ask:
                    self._latest_price = (float(bid) + float(ask)) / 2

            def on_state(state_name: str):
                log.info("Market stream state: %s", state_name)

            self._data_stream = DataStream(
                api_client=self._api_client,
                contract_id_to_subscribe=self._contract_id,
                on_quote_callback=on_quote,
                on_state_change_callback=on_state,
            )

            await asyncio.to_thread(self._data_stream.start)
            log.info("Quote stream started for %s", self._contract_id)

        except Exception as e:
            log.error("Failed to start quote stream: %s", e)

    def _update_latest_price(self):
        """Update latest price from DataManager if available."""
        if self._data_manager:
            try:
                df = self._data_manager.get_chart_data(timeframe_seconds=60)
                if df is not None and not df.empty:
                    self._latest_price = float(df.iloc[-1].get('close', 0))
            except Exception:
                pass

    async def _get_positions(self) -> list[Position]:
        """Fetch current positions from API."""
        try:
            raw_positions = await asyncio.to_thread(
                self._api_client.search_positions,
                account_id=self._account_id)

            positions = []
            for p in raw_positions:
                qty = getattr(p, 'size', getattr(p, 'quantity', 0))
                positions.append(Position(
                    symbol=getattr(p, 'contract_id', self._contract_id),
                    quantity=qty,
                    avg_entry=getattr(p, 'average_price',
                                      getattr(p, 'avg_price', 0)),
                    unrealized_pnl=getattr(p, 'unrealized_pnl',
                                           getattr(p, 'pnl', 0)),
                    realized_pnl=getattr(p, 'realized_pnl', 0),
                ))
            return positions

        except Exception as e:
            log.error("Failed to fetch positions: %s", e)
            return []

    async def _start_user_stream(self):
        """Start user data stream for order/position updates."""
        if self._user_stream is not None:
            return

        try:
            from tsxapipy import UserHubStream

            def on_order(order_data: dict):
                order_id = str(order_data.get('id', ''))
                self._order_updates[order_id] = order_data
                log.info("Order update: %s", order_data)

            def on_state(state_name: str):
                log.info("User stream state: %s", state_name)

            self._user_stream = UserHubStream(
                api_client=self._api_client,
                account_id_to_watch=self._account_id,
                on_order_update=on_order,
                on_state_change_callback=on_state,
            )

            await asyncio.to_thread(self._user_stream.start)
            log.info("User stream started for account %d",
                     self._account_id)

        except Exception as e:
            log.error("Failed to start user stream: %s", e)
