"""
Challenge state tracker and position sizer.

Tracks account state relative to challenge rules and computes
position sizes using the appropriate strategy:

  Phase 1 (challenge):  frac_2pct — fixed fractional 2% of equity
  Phase 2 (funded):     dd_frac_40pct — 40% of remaining drawdown buffer

Monitors:
  - Profit target progress
  - Max drawdown (from initial balance)
  - Daily drawdown
  - Trading day count
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import sim_config as cfg

log = logging.getLogger("bot.tracker")


class Phase(Enum):
    CHALLENGE = "CHALLENGE"
    FUNDED = "FUNDED"


class ChallengeStatus(Enum):
    ACTIVE = "ACTIVE"
    PASSED = "PASSED"
    FAILED_DRAWDOWN = "FAILED_DRAWDOWN"
    FAILED_DAILY_DD = "FAILED_DAILY_DD"
    FAILED_TIME = "FAILED_TIME"


@dataclass
class DailyRecord:
    date: str
    start_equity: float
    end_equity: float
    pnl: float
    trades: int
    peak_equity: float
    max_drawdown: float


class ChallengeTracker:
    """
    Tracks challenge/funded account state and computes position sizes.

    Usage:
        tracker = ChallengeTracker(phase=Phase.CHALLENGE)
        size = tracker.compute_size(risk_per_contract=500.0)
        tracker.record_trade(pnl=250.0)
        tracker.check_limits()
    """

    def __init__(self, phase: Phase = Phase.CHALLENGE,
                 account_size: float = None):
        self.phase = phase
        self.initial_equity = account_size or cfg.ACCOUNT_SIZE
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.realized_pnl = 0.0

        # Daily tracking
        self.daily_start_equity = self.equity
        self.daily_pnl = 0.0
        self.trading_days = 0
        self.trades_today = 0
        self.daily_records: list[DailyRecord] = []

        # Challenge limits
        if phase == Phase.CHALLENGE:
            self.profit_target = self.initial_equity * (
                1 + cfg.PROFIT_TARGET_PCT)
            self.max_dd_level = self.initial_equity * (
                1 - cfg.MAX_DRAWDOWN_PCT)
            self.max_daily_dd = self.initial_equity * cfg.MAX_DAILY_DRAWDOWN_PCT
        else:
            self.profit_target = float('inf')  # No target when funded
            self.max_dd_level = self.initial_equity * (
                1 - cfg.FUNDED_MAX_DRAWDOWN_PCT)
            self.max_daily_dd = self.initial_equity * cfg.MAX_DAILY_DRAWDOWN_PCT

        self.status = ChallengeStatus.ACTIVE
        self.total_trades = 0

        log.info("Tracker init  phase=%s  equity=$%.0f  target=$%.0f  "
                 "dd_floor=$%.0f",
                 phase.value, self.equity, self.profit_target,
                 self.max_dd_level)

    def on_day_open(self):
        """Call at session open each day."""
        self.daily_start_equity = self.equity
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.trading_days += 1
        log.info("Day %d open  equity=$%.0f  P&L=$%.0f  target_dist=$%.0f",
                 self.trading_days, self.equity, self.realized_pnl,
                 self.profit_target - self.equity)

    def on_day_close(self):
        """Call at session close."""
        record = DailyRecord(
            date=time.strftime("%Y-%m-%d"),
            start_equity=self.daily_start_equity,
            end_equity=self.equity,
            pnl=self.daily_pnl,
            trades=self.trades_today,
            peak_equity=self.peak_equity,
            max_drawdown=(self.initial_equity - min(
                self.equity, self.daily_start_equity)) / self.initial_equity,
        )
        self.daily_records.append(record)
        log.info("Day %d close  equity=$%.0f  daily_pnl=$%.0f  "
                 "trades=%d",
                 self.trading_days, self.equity, self.daily_pnl,
                 self.trades_today)

    def compute_size(self, risk_per_contract: float,
                      min_contracts: int = 1,
                      max_contracts: int = 50) -> int:
        """
        Compute position size in contracts.

        Args:
            risk_per_contract: Dollar risk per contract (stop distance * point value)
            min_contracts: Minimum position size
            max_contracts: Maximum position size (platform/risk limit)

        Returns:
            Number of contracts to trade
        """
        if self.status != ChallengeStatus.ACTIVE:
            return 0
        if risk_per_contract <= 0:
            return 0

        if self.phase == Phase.CHALLENGE:
            # frac_2pct: risk 2% of current equity
            risk_budget = self.equity * cfg.FRAC_2PCT["risk_per_trade"]
        else:
            # dd_frac_40pct: risk 40% of remaining drawdown buffer
            buffer = self.equity - self.max_dd_level
            if buffer <= 0:
                return 0
            risk_budget = buffer * cfg.DD_FRAC_40PCT["dd_fraction"]

        # Check daily drawdown headroom
        daily_dd_remaining = self.max_daily_dd - abs(min(0, self.daily_pnl))
        risk_budget = min(risk_budget, daily_dd_remaining)

        if risk_budget <= 0:
            return 0

        contracts = math.floor(risk_budget / risk_per_contract)
        contracts = max(min_contracts, min(contracts, max_contracts))

        log.info("SIZE  budget=$%.0f  risk/ct=$%.0f  → %d contracts"
                 "  [%s]",
                 risk_budget, risk_per_contract, contracts,
                 self.phase.value)

        return contracts

    def record_trade(self, pnl: float):
        """Record a completed trade's PnL."""
        self.equity += pnl
        self.realized_pnl += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        self.trades_today += 1
        self.peak_equity = max(self.peak_equity, self.equity)

        log.info("TRADE  pnl=$%.0f  equity=$%.0f  daily=$%.0f  "
                 "total_pnl=$%.0f",
                 pnl, self.equity, self.daily_pnl, self.realized_pnl)

        self.check_limits()

    def check_limits(self) -> ChallengeStatus:
        """Check if any challenge limits have been breached."""
        if self.status != ChallengeStatus.ACTIVE:
            return self.status

        # Max drawdown from initial balance
        if self.equity <= self.max_dd_level:
            self.status = ChallengeStatus.FAILED_DRAWDOWN
            log.warning("FAILED — max drawdown hit  equity=$%.0f  "
                        "floor=$%.0f",
                        self.equity, self.max_dd_level)
            return self.status

        # Daily drawdown
        if self.daily_pnl <= -self.max_daily_dd:
            self.status = ChallengeStatus.FAILED_DAILY_DD
            log.warning("FAILED — daily drawdown hit  daily_pnl=$%.0f  "
                        "limit=$%.0f",
                        self.daily_pnl, -self.max_daily_dd)
            return self.status

        # Profit target (challenge only)
        if self.phase == Phase.CHALLENGE:
            if self.equity >= self.profit_target and \
               self.trading_days >= cfg.MIN_TRADING_DAYS:
                self.status = ChallengeStatus.PASSED
                log.info("PASSED — profit target reached  equity=$%.0f  "
                         "target=$%.0f  days=%d",
                         self.equity, self.profit_target, self.trading_days)
                return self.status

            if self.trading_days >= cfg.MAX_TRADING_DAYS:
                self.status = ChallengeStatus.FAILED_TIME
                log.warning("FAILED — time limit  days=%d",
                            self.trading_days)
                return self.status

        return self.status

    def can_trade(self) -> bool:
        """Check if we're allowed to take another trade."""
        if self.status != ChallengeStatus.ACTIVE:
            return False

        # Don't trade if daily DD is close to limit
        daily_dd_used = abs(min(0, self.daily_pnl))
        if daily_dd_used >= self.max_daily_dd * 0.90:
            log.info("Daily DD 90%% used ($%.0f / $%.0f) — no more trades",
                     daily_dd_used, self.max_daily_dd)
            return False

        return True

    def get_summary(self) -> dict:
        dd_from_initial = (self.initial_equity - self.equity) / \
            self.initial_equity if self.equity < self.initial_equity else 0
        dd_from_peak = (self.peak_equity - self.equity) / \
            self.initial_equity if self.equity < self.peak_equity else 0

        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "equity": self.equity,
            "initial_equity": self.initial_equity,
            "realized_pnl": self.realized_pnl,
            "daily_pnl": self.daily_pnl,
            "trading_days": self.trading_days,
            "total_trades": self.total_trades,
            "dd_from_initial": dd_from_initial,
            "dd_from_peak": dd_from_peak,
            "profit_target": self.profit_target,
            "max_dd_level": self.max_dd_level,
            "distance_to_target": self.profit_target - self.equity,
            "distance_to_dd": self.equity - self.max_dd_level,
        }

    def should_request_payout(self) -> bool:
        """For funded phase: check if we should request a payout."""
        if self.phase != Phase.FUNDED:
            return False
        gross = max(0, self.equity - self.initial_equity)
        return gross >= cfg.FUNDED_PAYOUT_THRESHOLD
