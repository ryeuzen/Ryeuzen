#!/usr/bin/env python3
"""
Trading bot orchestrator.

Wires together:
  - BrokerAdapter (order execution)
  - ORBStrategy (signal generation)
  - ChallengeTracker (position sizing + state management)

Runs an async loop that:
  1. Waits for session open (09:30 ET)
  2. Feeds bars to ORB strategy
  3. On breakout signal → sizes via tracker → submits bracket order
  4. Monitors stop/target fills
  5. Records trade result
  6. Repeats until challenge passed/failed

Usage:
  # Paper mode (dry run):
  python3 bot.py --paper

  # Live mode (requires broker config):
  python3 bot.py --live --symbol MNQ --broker tradovate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from broker import (
    BrokerAdapter, PaperBroker, OrderRequest, OrderFill,
    Side, OrderType, OrderStatus, Bar,
)
from orb_strategy import ORBStrategy, ORBConfig, ORBSignal, ORBState
from challenge_tracker import ChallengeTracker, Phase, ChallengeStatus
import sim_config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")

ET = ZoneInfo("America/New_York")

# ── NQ/MNQ contract specs ─────────────────────────────────────────
CONTRACTS = {
    "NQ": {"tick_size": 0.25, "tick_value": 5.00, "point_value": 20.00},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50, "point_value": 2.00},
    "ES": {"tick_size": 0.25, "tick_value": 12.50, "point_value": 50.00},
    "MES": {"tick_size": 0.25, "tick_value": 1.25, "point_value": 5.00},
}


class TradingBot:
    """
    Main bot orchestrator. Runs one session at a time.
    """

    def __init__(self, broker: BrokerAdapter, symbol: str = "MNQ",
                 phase: Phase = Phase.CHALLENGE,
                 orb_config: ORBConfig = None):
        self.broker = broker
        self.symbol = symbol
        self.phase = phase

        spec = CONTRACTS.get(symbol, CONTRACTS["MNQ"])
        self.tick_size = spec["tick_size"]
        self.tick_value = spec["tick_value"]
        self.point_value = spec["point_value"]

        self.orb = ORBStrategy(orb_config or ORBConfig(
            tick_size=self.tick_size))
        self.tracker = ChallengeTracker(phase=phase)

        self._running = False
        self._current_order_ids: list[str] = []
        self._trade_log: list[dict] = []

    async def run(self):
        """Main bot loop. Runs until challenge resolved or interrupted."""
        self._running = True
        await self.broker.connect()

        log.info("Bot started  symbol=%s  phase=%s", self.symbol,
                 self.phase.value)
        log.info("Contract: tick=%.2f  tick_val=$%.2f  pt_val=$%.2f",
                 self.tick_size, self.tick_value, self.point_value)

        try:
            while self._running and \
                  self.tracker.status == ChallengeStatus.ACTIVE:
                await self._run_session()
                if self.tracker.status != ChallengeStatus.ACTIVE:
                    break
                # Wait for next session
                await self._wait_for_next_session()
        except asyncio.CancelledError:
            log.info("Bot cancelled")
        finally:
            # Flatten any open positions
            await self.broker.flatten(self.symbol)
            await self.broker.disconnect()
            self._print_summary()

    async def _run_session(self):
        """Run a single trading session (one day)."""
        self.tracker.on_day_open()
        self.orb.on_session_open()

        log.info("─" * 50)
        log.info("SESSION %d  equity=$%.0f",
                 self.tracker.trading_days, self.tracker.equity)
        log.info("─" * 50)

        bar_interval = 1  # 1-minute bars

        try:
            bars = await self.broker.get_bars(
                self.symbol, bar_interval, count=500)

            if bars:
                # Process historical bars for range building
                for bar in bars:
                    await self._process_bar(bar)

            # Stream live bars
            await self._stream_session(bar_interval)

        except Exception as e:
            log.error("Session error: %s", e)

        self.orb.on_session_close()
        self.tracker.on_day_close()

    async def _stream_session(self, interval: int):
        """
        Stream bars for the session duration.
        For paper mode, this simulates bar-by-bar.
        For live, this connects to the broker's bar stream.
        """
        # Define callback
        async def on_bar(bar: Bar):
            await self._process_bar(bar)

        # Stream until session close or done
        await self.broker.stream_bars(self.symbol, interval, on_bar)

    async def _process_bar(self, bar: Bar):
        """Process a single bar through the strategy."""
        if not self.tracker.can_trade():
            return

        if self.orb.state == ORBState.DONE_FOR_DAY:
            return

        signal = self.orb.on_bar(bar)

        if signal is not None:
            await self._execute_signal(signal)

    async def _execute_signal(self, signal: ORBSignal):
        """Size and execute a trade from an ORB signal."""
        # Compute risk per contract
        risk_per_contract = signal.risk_points * self.point_value

        if risk_per_contract <= 0:
            log.warning("Invalid risk_per_contract: %.2f", risk_per_contract)
            return

        # Get position size from tracker
        contracts = self.tracker.compute_size(
            risk_per_contract=risk_per_contract)

        if contracts <= 0:
            log.info("Size = 0 contracts — skipping")
            self.orb.on_trade_closed()
            return

        log.info("EXECUTING  %s %d %s  entry=%.2f  stop=%.2f  "
                 "target=%.2f  risk/ct=$%.0f",
                 signal.side.value, contracts, self.symbol,
                 signal.entry_price, signal.stop_price,
                 signal.target_price, risk_per_contract)

        # Submit entry order
        entry_fill = await self.broker.submit_order(OrderRequest(
            symbol=self.symbol,
            side=signal.side,
            quantity=contracts,
            order_type=OrderType.MARKET,
            tag="orb_entry",
        ))

        if entry_fill.status != OrderStatus.FILLED:
            log.warning("Entry rejected: %s", entry_fill.status.value)
            self.orb.on_trade_closed()
            return

        # Submit stop and target as OCO (or manage manually)
        exit_side = Side.SELL if signal.side == Side.BUY else Side.BUY

        # Stop order
        stop_fill = await self._manage_exit(
            entry_fill, signal, exit_side, contracts)

        # Record result
        if stop_fill:
            if signal.side == Side.BUY:
                pnl = (stop_fill.fill_price - entry_fill.fill_price) * \
                    contracts * self.point_value
            else:
                pnl = (entry_fill.fill_price - stop_fill.fill_price) * \
                    contracts * self.point_value

            self.tracker.record_trade(pnl)

            trade_record = {
                "day": self.tracker.trading_days,
                "side": signal.side.value,
                "contracts": contracts,
                "entry": entry_fill.fill_price,
                "exit": stop_fill.fill_price,
                "pnl": pnl,
                "equity": self.tracker.equity,
                "tag": stop_fill.tag,
            }
            self._trade_log.append(trade_record)

            log.info("CLOSED  pnl=$%.0f  equity=$%.0f  tag=%s",
                     pnl, self.tracker.equity, stop_fill.tag)

        self.orb.on_trade_closed()

    async def _manage_exit(self, entry: OrderFill, signal: ORBSignal,
                            exit_side: Side,
                            contracts: int) -> Optional[OrderFill]:
        """
        Monitor position and exit at stop or target.

        In live mode, you'd submit OCO bracket orders.
        This implementation polls price for compatibility with
        all broker adapters.
        """
        stop = signal.stop_price
        target = signal.target_price
        is_long = signal.side == Side.BUY

        log.info("Monitoring exit  stop=%.2f  target=%.2f", stop, target)

        while self._running:
            try:
                price = await self.broker.get_tick(self.symbol)
                if price == 0:
                    await asyncio.sleep(0.5)
                    continue

                # Check stop
                if (is_long and price <= stop) or \
                   (not is_long and price >= stop):
                    fill = await self.broker.submit_order(OrderRequest(
                        symbol=self.symbol, side=exit_side,
                        quantity=contracts, tag="orb_stop"))
                    return fill

                # Check target
                if (is_long and price >= target) or \
                   (not is_long and price <= target):
                    fill = await self.broker.submit_order(OrderRequest(
                        symbol=self.symbol, side=exit_side,
                        quantity=contracts, tag="orb_target"))
                    return fill

                await asyncio.sleep(0.25)

            except Exception as e:
                log.error("Exit monitor error: %s", e)
                # Emergency flatten
                return await self.broker.flatten(self.symbol)

        return None

    async def _wait_for_next_session(self):
        """Wait until next trading session opens."""
        now = datetime.now(ET)
        next_open = now.replace(
            hour=9, minute=30, second=0, microsecond=0)
        if now >= next_open:
            next_open += timedelta(days=1)
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)

        wait_seconds = (next_open - now).total_seconds()
        log.info("Next session: %s ET (%.0f hours)",
                 next_open.strftime("%Y-%m-%d %H:%M"), wait_seconds / 3600)

        await asyncio.sleep(wait_seconds)

    def stop(self):
        """Graceful shutdown."""
        self._running = False

    def _print_summary(self):
        summary = self.tracker.get_summary()
        log.info("=" * 50)
        log.info("  BOT SUMMARY")
        log.info("=" * 50)
        log.info("  Status:       %s", summary["status"])
        log.info("  Phase:        %s", summary["phase"])
        log.info("  Equity:       $%.0f", summary["equity"])
        log.info("  Total PnL:    $%.0f", summary["realized_pnl"])
        log.info("  Trading days: %d", summary["trading_days"])
        log.info("  Total trades: %d", summary["total_trades"])
        log.info("  DD from init: %.2f%%",
                 summary["dd_from_initial"] * 100)
        log.info("  Dist to tgt:  $%.0f", summary["distance_to_target"])
        log.info("  Dist to DD:   $%.0f", summary["distance_to_dd"])
        log.info("=" * 50)

        if self._trade_log:
            log.info("  Trade log:")
            for t in self._trade_log:
                log.info("    Day %d  %s %d  %.2f→%.2f  pnl=$%.0f  [%s]",
                         t["day"], t["side"], t["contracts"],
                         t["entry"], t["exit"], t["pnl"], t["tag"])


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ORB Trading Bot for Prop Firm Challenges")
    parser.add_argument("--broker", default="paper",
                        choices=["paper", "topstep"],
                        help="Broker adapter (default: paper)")
    parser.add_argument("--symbol", default="MNQ",
                        choices=list(CONTRACTS.keys()),
                        help="Instrument to trade (default: MNQ)")
    parser.add_argument("--phase", default="challenge",
                        choices=["challenge", "funded"],
                        help="Challenge phase (default: challenge)")
    parser.add_argument("--range-minutes", type=int, default=5,
                        help="Opening range duration in minutes")
    parser.add_argument("--rr", type=float, default=0.5,
                        help="Reward:Risk ratio (default: 0.5)")
    parser.add_argument("--max-range", type=float, default=30.0,
                        help="Max range size in points to trade")
    parser.add_argument("--min-range", type=float, default=3.0,
                        help="Min range size in points to trade")
    parser.add_argument("--contract-id", default=None,
                        help="TopstepX contract ID (e.g. CON.F.US.MNQ.M25)")
    parser.add_argument("--account-id", type=int, default=None,
                        help="TopstepX account ID (auto-detects if omitted)")
    return parser.parse_args()


async def main():
    args = parse_args()

    orb_config = ORBConfig(
        range_minutes=args.range_minutes,
        reward_risk_ratio=args.rr,
        max_range_points=args.max_range,
        min_range_points=args.min_range,
        tick_size=CONTRACTS[args.symbol]["tick_size"],
    )

    phase = Phase.FUNDED if args.phase == "funded" else Phase.CHALLENGE

    if args.broker == "topstep":
        from broker_topstep import TopstepBroker
        broker = TopstepBroker(
            contract_id=args.contract_id,
            account_id=args.account_id,
        )
    else:
        broker = PaperBroker(
            initial_equity=cfg.ACCOUNT_SIZE,
            tick_value=CONTRACTS[args.symbol]["tick_value"],
            tick_size=CONTRACTS[args.symbol]["tick_size"],
        )

    bot = TradingBot(
        broker=broker,
        symbol=args.symbol,
        phase=phase,
        orb_config=orb_config,
    )

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.stop)

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
