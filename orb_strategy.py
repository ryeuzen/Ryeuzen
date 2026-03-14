"""
Opening Range Breakout (ORB) strategy engine.

Defines the opening range as the high/low of the first N minutes
after market open. Enters on breakout, with stop at opposite side
of range and target at RR * range.

For NQ/MNQ:
  - Regular session opens at 09:30 ET
  - Opening range: first 5 minutes (09:30-09:35)
  - Breakout entry: price exceeds range high/low
  - Stop: opposite end of range
  - Target: RR * range_size from entry

Position sizing is delegated to the challenge tracker (frac_2pct
or dd_frac_40pct) — this module only generates signals and
computes stop/target levels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from broker import Bar, Side

log = logging.getLogger("bot.orb")


class ORBState(Enum):
    WAITING_FOR_OPEN = "WAITING_FOR_OPEN"
    BUILDING_RANGE = "BUILDING_RANGE"
    WATCHING_FOR_BREAKOUT = "WATCHING_FOR_BREAKOUT"
    IN_TRADE = "IN_TRADE"
    DONE_FOR_DAY = "DONE_FOR_DAY"


@dataclass
class ORBLevels:
    range_high: float
    range_low: float
    range_size: float          # high - low
    long_entry: float          # range_high + buffer
    short_entry: float         # range_low - buffer
    long_stop: float           # range_low
    short_stop: float          # range_high
    long_target: float         # entry + RR * range_size
    short_target: float        # entry - RR * range_size


@dataclass
class ORBSignal:
    side: Side
    entry_price: float
    stop_price: float
    target_price: float
    risk_points: float         # Distance entry to stop in points
    reward_points: float       # Distance entry to target in points
    range_size: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ORBConfig:
    range_minutes: int = 5           # Duration of opening range
    reward_risk_ratio: float = 0.5   # RR = 0.5 (target is half the stop)
    entry_buffer_ticks: int = 2      # Ticks beyond range for entry
    tick_size: float = 0.25          # NQ tick size
    max_range_points: float = 30.0   # Skip if range too wide
    min_range_points: float = 3.0    # Skip if range too narrow
    max_trades_per_day: int = 1      # One ORB trade per session
    session_open_hour: int = 9       # ET
    session_open_minute: int = 30
    session_close_hour: int = 16     # ET
    session_close_minute: int = 0


class ORBStrategy:
    """
    Stateful ORB strategy engine. Feed it bars and it produces signals.

    Usage:
        orb = ORBStrategy()
        orb.on_session_open()          # Call at 09:30 ET
        for bar in bars:
            signal = orb.on_bar(bar)   # Returns ORBSignal or None
            if signal:
                # Execute trade
    """

    def __init__(self, config: ORBConfig = None):
        self.cfg = config or ORBConfig()
        self.state = ORBState.WAITING_FOR_OPEN
        self.range_bars: list[Bar] = []
        self.levels: Optional[ORBLevels] = None
        self.trades_today = 0
        self._in_trade_side: Optional[Side] = None

    def on_session_open(self):
        """Call at market open to start building the range."""
        self.state = ORBState.BUILDING_RANGE
        self.range_bars = []
        self.levels = None
        self.trades_today = 0
        self._in_trade_side = None
        log.info("Session open — building %d-min opening range",
                 self.cfg.range_minutes)

    def on_session_close(self):
        """Call at session end."""
        self.state = ORBState.DONE_FOR_DAY
        log.info("Session closed")

    def on_bar(self, bar: Bar) -> Optional[ORBSignal]:
        """
        Process a new bar. Returns ORBSignal if a breakout is detected.

        Bar interval should be 1 minute for standard ORB.
        """
        if self.state == ORBState.BUILDING_RANGE:
            return self._build_range(bar)
        elif self.state == ORBState.WATCHING_FOR_BREAKOUT:
            return self._check_breakout(bar)
        return None

    def on_price(self, price: float, timestamp: float = None) -> Optional[ORBSignal]:
        """
        Process a tick/price update. For real-time breakout detection
        without waiting for bar close.
        """
        if self.state != ORBState.WATCHING_FOR_BREAKOUT:
            return None
        if self.levels is None:
            return None

        return self._evaluate_breakout(price, timestamp or time.time())

    def _build_range(self, bar: Bar) -> Optional[ORBSignal]:
        self.range_bars.append(bar)

        if len(self.range_bars) >= self.cfg.range_minutes:
            # Compute range
            high = max(b.high for b in self.range_bars)
            low = min(b.low for b in self.range_bars)
            size = high - low

            log.info("Opening range: H=%.2f  L=%.2f  Size=%.2f pts",
                     high, low, size)

            # Validate range
            if size > self.cfg.max_range_points:
                log.info("Range too wide (%.2f > %.2f) — skip today",
                         size, self.cfg.max_range_points)
                self.state = ORBState.DONE_FOR_DAY
                return None

            if size < self.cfg.min_range_points:
                log.info("Range too narrow (%.2f < %.2f) — skip today",
                         size, self.cfg.min_range_points)
                self.state = ORBState.DONE_FOR_DAY
                return None

            buffer = self.cfg.entry_buffer_ticks * self.cfg.tick_size
            rr = self.cfg.reward_risk_ratio

            self.levels = ORBLevels(
                range_high=high,
                range_low=low,
                range_size=size,
                long_entry=high + buffer,
                short_entry=low - buffer,
                long_stop=low,
                short_stop=high,
                long_target=high + buffer + (size + buffer) * rr,
                short_target=low - buffer - (size + buffer) * rr,
            )

            log.info("Levels set — Long entry=%.2f stop=%.2f tgt=%.2f",
                     self.levels.long_entry, self.levels.long_stop,
                     self.levels.long_target)
            log.info("             Short entry=%.2f stop=%.2f tgt=%.2f",
                     self.levels.short_entry, self.levels.short_stop,
                     self.levels.short_target)

            self.state = ORBState.WATCHING_FOR_BREAKOUT

        return None

    def _check_breakout(self, bar: Bar) -> Optional[ORBSignal]:
        """Check bar for breakout (using high/low of bar)."""
        if self.levels is None:
            return None

        # Check long breakout first (high of bar exceeds entry)
        if bar.high >= self.levels.long_entry:
            return self._evaluate_breakout(
                self.levels.long_entry, bar.timestamp)

        # Check short breakout
        if bar.low <= self.levels.short_entry:
            return self._evaluate_breakout(
                self.levels.short_entry, bar.timestamp)

        return None

    def _evaluate_breakout(self, price: float,
                            timestamp: float) -> Optional[ORBSignal]:
        if self.levels is None:
            return None
        if self.trades_today >= self.cfg.max_trades_per_day:
            return None

        lvl = self.levels

        if price >= lvl.long_entry:
            side = Side.BUY
            entry = lvl.long_entry
            stop = lvl.long_stop
            target = lvl.long_target
        elif price <= lvl.short_entry:
            side = Side.SELL
            entry = lvl.short_entry
            stop = lvl.short_stop
            target = lvl.short_target
        else:
            return None

        risk_pts = abs(entry - stop)
        reward_pts = abs(target - entry)

        signal = ORBSignal(
            side=side, entry_price=entry, stop_price=stop,
            target_price=target, risk_points=risk_pts,
            reward_points=reward_pts, range_size=lvl.range_size,
            timestamp=timestamp)

        self.trades_today += 1
        self.state = ORBState.IN_TRADE
        self._in_trade_side = side

        log.info("BREAKOUT %s  entry=%.2f  stop=%.2f  target=%.2f  "
                 "risk=%.2f  reward=%.2f",
                 side.value, entry, stop, target, risk_pts, reward_pts)

        return signal

    def on_trade_closed(self):
        """Call when the trade (stop/target/manual) is closed."""
        self._in_trade_side = None
        if self.trades_today >= self.cfg.max_trades_per_day:
            self.state = ORBState.DONE_FOR_DAY
            log.info("Max trades reached — done for day")
        else:
            self.state = ORBState.WATCHING_FOR_BREAKOUT

    def get_state_summary(self) -> dict:
        return {
            "state": self.state.value,
            "trades_today": self.trades_today,
            "range": {
                "high": self.levels.range_high if self.levels else None,
                "low": self.levels.range_low if self.levels else None,
                "size": self.levels.range_size if self.levels else None,
            },
            "in_trade": self._in_trade_side.value if self._in_trade_side else None,
        }
