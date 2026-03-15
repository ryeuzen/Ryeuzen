"""
Monte Carlo simulator for prop firm challenge strategies.

Two-phase approach:
  Phase 1 (challenge): Pass the challenge using zero-EV ORB with
    zone-based adaptive sizing. Zones adjust risk fraction based on
    equity position relative to target and DD floor.

  Phase 2 (funded): Once funded, maximize payout by risking a
    fraction of remaining drawdown buffer per trade (dd_frac).

Key modeling details:
  - Daily drawdown resets each session (tracked separately from overall DD)
  - Daily stop rule: halt trading for the day once X% of daily DD used
  - Zone transitions are checked before every trade
  - Hail Mary zone triggers near max DD for a last-chance burst
  - Profit target only checked after MIN_TRADING_DAYS

Combined EV = P(pass) * E[funded_payout] - challenge_fee
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sim_config as cfg


# ── Data classes ───────────────────────────────────────────────────

@dataclass
class TradeResult:
    pnl: float
    equity_after: float
    drawdown: float
    risk_amount: float
    won: bool
    zone: str


@dataclass
class ChallengeResult:
    passed: bool
    final_equity: float
    peak_equity: float
    max_drawdown: float
    max_daily_drawdown: float   # Worst single-day loss
    num_trades: int
    num_days: int
    terminal_zone: str          # Zone when challenge ended
    equity_curve: list[float] = field(default_factory=list)
    daily_pnls: list[float] = field(default_factory=list)
    zone_trades: dict = field(default_factory=dict)  # trades per zone


@dataclass
class FundedResult:
    final_equity: float
    peak_equity: float
    max_drawdown: float
    gross_profit: float
    payout: float
    num_trades: int
    blown: bool
    equity_curve: list[float] = field(default_factory=list)


# ── Zone classification ────────────────────────────────────────────

def classify_zone(equity: float, initial: float,
                  dd_floor: float, profit_target: float,
                  zones: dict = None) -> str:
    """
    Determine which sizing zone the account is in.

    Args:
        equity: Current equity
        initial: Starting account size
        dd_floor: Max drawdown level (equity at which challenge fails)
        profit_target: Equity level to pass challenge
        zones: Zone config dict (defaults to cfg.ZONE_SIZING)

    Returns:
        Zone name: "cruise", "normal", "caution", or "hail_mary"
    """
    if zones is None:
        zones = cfg.ZONE_SIZING

    total_buffer = initial - dd_floor              # Total DD budget
    remaining_buffer = equity - dd_floor           # What's left before death
    target_distance = profit_target - initial      # Total distance to target
    profit_progress = equity - initial             # How far toward target

    # Hail Mary: remaining buffer is ≤ threshold fraction of total buffer
    if total_buffer > 0:
        buffer_pct = remaining_buffer / total_buffer
        if buffer_pct <= zones["hail_mary_threshold"]:
            return "hail_mary"

    # Cruise: profit progress is ≥ threshold fraction of target distance
    if target_distance > 0 and profit_progress > 0:
        progress_pct = profit_progress / target_distance
        if progress_pct >= zones["cruise_threshold"]:
            return "cruise"

    # Caution: equity below initial (in drawdown)
    if equity < initial:
        return "caution"

    # Normal: everything else
    return "normal"


def get_zone_risk(zone: str, zones: dict = None) -> float:
    """Get risk fraction for a given zone."""
    if zones is None:
        zones = cfg.ZONE_SIZING
    return zones[f"risk_{zone}"]


# ── Challenge simulator ───────────────────────────────────────────

def simulate_challenge(rng: np.random.Generator,
                       win_rate: float = None,
                       reward_risk: float = None,
                       zones: dict = None,
                       trades_per_day: int = None,
                       record_curve: bool = False) -> ChallengeResult:
    """
    Simulate a single challenge attempt with zone-based sizing.

    All strategy parameters are inputs so the sim can be driven by
    any (win_rate, RR) combination without code changes.

    The daily drawdown is modeled properly:
      - Each day, day_start_equity is recorded
      - If (day_start_equity - equity) >= max_daily_dd → fail
      - daily_stop_pct causes early halt before limit is hit
      - Overall max DD is checked from initial balance (not peak)
    """
    if win_rate is None:
        win_rate = cfg.DEFAULT_WIN_RATE
    if reward_risk is None:
        reward_risk = cfg.DEFAULT_REWARD_RISK
    if zones is None:
        zones = cfg.ZONE_SIZING
    if trades_per_day is None:
        trades_per_day = cfg.TRADES_PER_DAY

    equity = float(cfg.ACCOUNT_SIZE)
    initial = float(cfg.ACCOUNT_SIZE)
    peak = equity
    profit_target = initial * (1.0 + cfg.PROFIT_TARGET_PCT)
    dd_floor = initial * (1.0 - cfg.MAX_DRAWDOWN_PCT)
    max_daily_dd = initial * cfg.MAX_DAILY_DRAWDOWN_PCT
    daily_stop_limit = max_daily_dd * zones["daily_stop_pct"]

    trades = 0
    worst_daily_dd = 0.0
    curve = [equity] if record_curve else []
    daily_pnls = []
    zone_trades = {"normal": 0, "cruise": 0, "caution": 0, "hail_mary": 0}

    for day in range(cfg.MAX_TRADING_DAYS):
        day_start_equity = equity
        day_pnl = 0.0
        day_stopped = False

        for t in range(trades_per_day):
            # ── Daily stop check ──
            # Stop trading for the day once we've used daily_stop_pct
            # of the daily DD limit. This preserves room for the daily
            # DD rule to not trigger.
            if day_pnl < 0 and abs(day_pnl) >= daily_stop_limit:
                day_stopped = True
                break

            # ── Zone classification ──
            zone = classify_zone(equity, initial, dd_floor,
                                 profit_target, zones)
            risk_pct = get_zone_risk(zone, zones)
            zone_trades[zone] += 1

            # ── Position sizing ──
            risk_amount = equity * risk_pct

            # Cap risk at remaining daily DD headroom
            daily_dd_remaining = max_daily_dd - abs(min(0.0, day_pnl))
            if daily_dd_remaining <= 0:
                break
            risk_amount = min(risk_amount, daily_dd_remaining * 0.95)

            # Floor: skip if risk amount too small to matter
            if risk_amount < 50:
                break

            # ── Trade outcome ──
            if rng.random() < win_rate:
                pnl = risk_amount * reward_risk
            else:
                pnl = -risk_amount

            equity += pnl
            day_pnl += pnl
            trades += 1
            peak = max(peak, equity)

            if record_curve:
                curve.append(equity)

            # ── Daily drawdown check (hard limit) ──
            daily_dd = day_start_equity - equity
            if daily_dd >= max_daily_dd:
                worst_daily_dd = max(worst_daily_dd, daily_dd)
                daily_pnls.append(day_pnl)
                zone = classify_zone(equity, initial, dd_floor,
                                     profit_target, zones)
                return ChallengeResult(
                    passed=False, final_equity=equity, peak_equity=peak,
                    max_drawdown=(initial - equity) / initial,
                    max_daily_drawdown=daily_dd / initial,
                    num_trades=trades, num_days=day + 1,
                    terminal_zone=zone, equity_curve=curve,
                    daily_pnls=daily_pnls, zone_trades=zone_trades)

            # ── Overall max drawdown check (from initial) ──
            if equity <= dd_floor:
                worst_daily_dd = max(worst_daily_dd,
                                     day_start_equity - equity)
                daily_pnls.append(day_pnl)
                zone = classify_zone(equity, initial, dd_floor,
                                     profit_target, zones)
                return ChallengeResult(
                    passed=False, final_equity=equity, peak_equity=peak,
                    max_drawdown=(initial - equity) / initial,
                    max_daily_drawdown=worst_daily_dd / initial,
                    num_trades=trades, num_days=day + 1,
                    terminal_zone=zone, equity_curve=curve,
                    daily_pnls=daily_pnls, zone_trades=zone_trades)

            # ── Profit target check (only after min trading days) ──
            if (day + 1) >= cfg.MIN_TRADING_DAYS and equity >= profit_target:
                worst_daily_dd = max(worst_daily_dd,
                                     max(0, day_start_equity - equity))
                daily_pnls.append(day_pnl)
                return ChallengeResult(
                    passed=True, final_equity=equity, peak_equity=peak,
                    max_drawdown=(peak - equity) / initial
                    if peak > equity else 0,
                    max_daily_drawdown=worst_daily_dd / initial,
                    num_trades=trades, num_days=day + 1,
                    terminal_zone="cruise", equity_curve=curve,
                    daily_pnls=daily_pnls, zone_trades=zone_trades)

        # End of day
        worst_daily_dd = max(worst_daily_dd,
                             max(0, day_start_equity - equity))
        daily_pnls.append(day_pnl)

    # Time expired
    zone = classify_zone(equity, initial, dd_floor, profit_target, zones)
    max_dd = (peak - equity) / initial if peak > equity else 0
    return ChallengeResult(
        passed=False, final_equity=equity, peak_equity=peak,
        max_drawdown=max_dd,
        max_daily_drawdown=worst_daily_dd / initial,
        num_trades=trades, num_days=cfg.MAX_TRADING_DAYS,
        terminal_zone=zone, equity_curve=curve,
        daily_pnls=daily_pnls, zone_trades=zone_trades)


# ── Flat (non-zone) challenge simulator for comparison ─────────────

def simulate_challenge_flat(rng: np.random.Generator,
                            win_rate: float = None,
                            reward_risk: float = None,
                            risk_pct: float = None,
                            trades_per_day: int = None,
                            record_curve: bool = False) -> ChallengeResult:
    """
    Simulate challenge with flat fixed-fractional sizing (no zones).
    Used as baseline comparison against zone-based approach.
    """
    if win_rate is None:
        win_rate = cfg.DEFAULT_WIN_RATE
    if reward_risk is None:
        reward_risk = cfg.DEFAULT_REWARD_RISK
    if risk_pct is None:
        risk_pct = cfg.FRAC_2PCT["risk_per_trade"]
    if trades_per_day is None:
        trades_per_day = cfg.TRADES_PER_DAY

    equity = float(cfg.ACCOUNT_SIZE)
    initial = float(cfg.ACCOUNT_SIZE)
    peak = equity
    profit_target = initial * (1.0 + cfg.PROFIT_TARGET_PCT)
    dd_floor = initial * (1.0 - cfg.MAX_DRAWDOWN_PCT)
    max_daily_dd = initial * cfg.MAX_DAILY_DRAWDOWN_PCT

    trades = 0
    worst_daily_dd = 0.0
    curve = [equity] if record_curve else []
    daily_pnls = []

    for day in range(cfg.MAX_TRADING_DAYS):
        day_start_equity = equity
        day_pnl = 0.0

        for _ in range(trades_per_day):
            risk_amount = equity * risk_pct

            if rng.random() < win_rate:
                pnl = risk_amount * reward_risk
            else:
                pnl = -risk_amount

            equity += pnl
            day_pnl += pnl
            trades += 1
            peak = max(peak, equity)

            if record_curve:
                curve.append(equity)

            # Daily DD check
            daily_dd = day_start_equity - equity
            if daily_dd >= max_daily_dd:
                worst_daily_dd = max(worst_daily_dd, daily_dd)
                daily_pnls.append(day_pnl)
                return ChallengeResult(
                    passed=False, final_equity=equity, peak_equity=peak,
                    max_drawdown=(initial - equity) / initial,
                    max_daily_drawdown=daily_dd / initial,
                    num_trades=trades, num_days=day + 1,
                    terminal_zone="n/a", equity_curve=curve,
                    daily_pnls=daily_pnls, zone_trades={})

            # Overall DD check
            if equity <= dd_floor:
                worst_daily_dd = max(worst_daily_dd,
                                     day_start_equity - equity)
                daily_pnls.append(day_pnl)
                return ChallengeResult(
                    passed=False, final_equity=equity, peak_equity=peak,
                    max_drawdown=(initial - equity) / initial,
                    max_daily_drawdown=worst_daily_dd / initial,
                    num_trades=trades, num_days=day + 1,
                    terminal_zone="n/a", equity_curve=curve,
                    daily_pnls=daily_pnls, zone_trades={})

            # Profit target (after min days)
            if (day + 1) >= cfg.MIN_TRADING_DAYS and equity >= profit_target:
                worst_daily_dd = max(worst_daily_dd,
                                     max(0, day_start_equity - equity))
                daily_pnls.append(day_pnl)
                return ChallengeResult(
                    passed=True, final_equity=equity, peak_equity=peak,
                    max_drawdown=(peak - equity) / initial
                    if peak > equity else 0,
                    max_daily_drawdown=worst_daily_dd / initial,
                    num_trades=trades, num_days=day + 1,
                    terminal_zone="n/a", equity_curve=curve,
                    daily_pnls=daily_pnls, zone_trades={})

        worst_daily_dd = max(worst_daily_dd,
                             max(0, day_start_equity - equity))
        daily_pnls.append(day_pnl)

    max_dd = (peak - equity) / initial if peak > equity else 0
    return ChallengeResult(
        passed=False, final_equity=equity, peak_equity=peak,
        max_drawdown=max_dd,
        max_daily_drawdown=worst_daily_dd / initial,
        num_trades=trades, num_days=cfg.MAX_TRADING_DAYS,
        terminal_zone="n/a", equity_curve=curve,
        daily_pnls=daily_pnls, zone_trades={})


# ── Funded account simulator ──────────────────────────────────────

def simulate_funded(rng: np.random.Generator,
                    win_rate: float = None,
                    reward_risk: float = None,
                    dd_fraction: float = None,
                    record_curve: bool = False) -> FundedResult:
    """
    Simulate funded account using dd_frac sizing.

    Sizing: risk = dd_fraction * remaining_buffer
    where remaining_buffer = max_dd_limit - current_drawdown_from_peak.

    The aggressive sizing extracts maximum payout before the
    inevitable blowup on a zero-EV strategy.
    """
    if win_rate is None:
        win_rate = cfg.DEFAULT_WIN_RATE
    if reward_risk is None:
        reward_risk = cfg.DEFAULT_REWARD_RISK
    if dd_fraction is None:
        dd_fraction = cfg.DD_FRAC_40PCT["dd_fraction"]

    equity = float(cfg.ACCOUNT_SIZE)
    initial = float(cfg.ACCOUNT_SIZE)
    peak = equity
    max_dd_dollars = initial * cfg.FUNDED_MAX_DRAWDOWN_PCT
    dd_floor = initial * (1.0 - cfg.FUNDED_MAX_DRAWDOWN_PCT)

    trades = 0
    high_water = equity
    curve = [equity] if record_curve else []

    for _ in range(cfg.FUNDED_MAX_TRADES):
        # Remaining drawdown buffer
        current_dd = max(0.0, high_water - equity)
        remaining_buffer = max_dd_dollars - current_dd

        if remaining_buffer <= 0:
            break

        risk_amount = remaining_buffer * dd_fraction

        if risk_amount < 100:
            break

        if rng.random() < win_rate:
            pnl = risk_amount * reward_risk
        else:
            pnl = -risk_amount

        equity += pnl
        trades += 1
        high_water = max(high_water, equity)
        peak = max(peak, equity)

        if record_curve:
            curve.append(equity)

        if equity <= dd_floor:
            gross = max(0.0, peak - initial)
            return FundedResult(
                final_equity=equity, peak_equity=peak,
                max_drawdown=(initial - equity) / initial,
                gross_profit=gross,
                payout=gross * cfg.FUNDED_PAYOUT_SPLIT,
                num_trades=trades, blown=True,
                equity_curve=curve)

    gross = max(0.0, equity - initial)
    return FundedResult(
        final_equity=equity, peak_equity=peak,
        max_drawdown=(peak - equity) / initial if peak > equity else 0,
        gross_profit=gross,
        payout=gross * cfg.FUNDED_PAYOUT_SPLIT,
        num_trades=trades, blown=False,
        equity_curve=curve)


# ── Full Monte Carlo ──────────────────────────────────────────────

def run_monte_carlo(n: int = None, seed: int = None,
                    win_rate: float = None,
                    reward_risk: float = None,
                    zones: dict = None,
                    compare_flat: bool = True,
                    verbose: bool = True) -> dict:
    """
    Run full two-phase Monte Carlo simulation.

    Simulates N challenge attempts with zone-based sizing, then
    funded accounts for those that pass. Optionally runs the same
    sim with flat sizing for head-to-head comparison.

    All strategy stats (win_rate, reward_risk, trade_frequency) are
    inputs — plug in different strategy characteristics without
    rewriting anything.

    Returns:
        Dict with challenge stats, funded stats, combined EV,
        and optionally the flat-sizing baseline comparison.
    """
    if n is None:
        n = cfg.MC_SIMULATIONS
    if seed is None:
        seed = cfg.MC_SEED
    if win_rate is None:
        win_rate = cfg.DEFAULT_WIN_RATE
    if reward_risk is None:
        reward_risk = cfg.DEFAULT_REWARD_RISK
    if zones is None:
        zones = cfg.ZONE_SIZING

    rng = np.random.default_rng(seed)

    # ── Phase 1: Zone-based challenge ──
    zone_results = []
    for _ in range(n):
        r = simulate_challenge(rng, win_rate=win_rate,
                               reward_risk=reward_risk, zones=zones)
        zone_results.append(r)

    zone_passed = [r for r in zone_results if r.passed]
    zone_pass_rate = len(zone_passed) / n

    # ── Phase 2: Funded accounts (zone passes) ──
    zone_funded = []
    zone_payouts = []
    for _ in range(len(zone_passed)):
        r = simulate_funded(rng, win_rate=win_rate,
                            reward_risk=reward_risk)
        zone_funded.append(r)
        zone_payouts.append(r.payout)

    zone_payouts = np.array(zone_payouts) if zone_payouts else np.array([0.0])

    # ── Flat baseline (same seed for fair comparison) ──
    flat_stats = None
    if compare_flat:
        rng_flat = np.random.default_rng(seed)
        flat_results = []
        for _ in range(n):
            r = simulate_challenge_flat(rng_flat, win_rate=win_rate,
                                        reward_risk=reward_risk)
            flat_results.append(r)

        flat_passed = [r for r in flat_results if r.passed]
        flat_pass_rate = len(flat_passed) / n

        flat_funded = []
        flat_payouts_list = []
        for _ in range(len(flat_passed)):
            r = simulate_funded(rng_flat, win_rate=win_rate,
                                reward_risk=reward_risk)
            flat_funded.append(r)
            flat_payouts_list.append(r.payout)

        flat_payouts = np.array(flat_payouts_list) if flat_payouts_list \
            else np.array([0.0])

        flat_avg_payout = float(np.mean(flat_payouts))
        flat_ev = flat_pass_rate * flat_avg_payout
        flat_net_ev = flat_ev - cfg.CHALLENGE_FEE

        # Failure mode breakdown for flat
        flat_fail_dd = sum(1 for r in flat_results
                           if not r.passed and r.max_drawdown >= cfg.MAX_DRAWDOWN_PCT - 0.001)
        flat_fail_daily = sum(1 for r in flat_results
                              if not r.passed and r.max_daily_drawdown >= cfg.MAX_DAILY_DRAWDOWN_PCT - 0.001
                              and r.max_drawdown < cfg.MAX_DRAWDOWN_PCT - 0.001)
        flat_fail_time = n - len(flat_passed) - flat_fail_dd - flat_fail_daily

        flat_stats = {
            "pass_rate": flat_pass_rate,
            "pass_count": len(flat_passed),
            "avg_payout": flat_avg_payout,
            "ev_per_attempt": flat_ev,
            "net_ev": flat_net_ev,
            "roi": flat_net_ev / cfg.CHALLENGE_FEE if cfg.CHALLENGE_FEE > 0 else 0,
            "fail_overall_dd": flat_fail_dd,
            "fail_daily_dd": flat_fail_daily,
            "fail_time": max(0, flat_fail_time),
        }

    # ── Compute zone stats ──
    avg_payout = float(np.mean(zone_payouts))
    median_payout = float(np.median(zone_payouts))
    ev_per_attempt = zone_pass_rate * avg_payout
    net_ev = ev_per_attempt - cfg.CHALLENGE_FEE

    # Payout percentiles
    if len(zone_payouts) > 1:
        pctiles = {f"p{p}": float(np.percentile(zone_payouts, p))
                   for p in [10, 25, 50, 75, 90, 99]}
        payout_std = float(np.std(zone_payouts))
    else:
        pctiles = {f"p{p}": 0.0 for p in [10, 25, 50, 75, 90, 99]}
        payout_std = 0.0

    # Funded stats
    if zone_funded:
        payout_rate = sum(1 for r in zone_funded if r.payout > 0) / len(zone_funded)
        blow_rate = sum(1 for r in zone_funded if r.blown) / len(zone_funded)
        avg_funded_trades = float(np.mean([r.num_trades for r in zone_funded]))
    else:
        payout_rate = blow_rate = avg_funded_trades = 0

    # Failure mode breakdown
    fail_dd = sum(1 for r in zone_results
                  if not r.passed and r.max_drawdown >= cfg.MAX_DRAWDOWN_PCT - 0.001)
    fail_daily = sum(1 for r in zone_results
                     if not r.passed and r.max_daily_drawdown >= cfg.MAX_DAILY_DRAWDOWN_PCT - 0.001
                     and r.max_drawdown < cfg.MAX_DRAWDOWN_PCT - 0.001)
    fail_time = n - len(zone_passed) - fail_dd - fail_daily

    # Zone utilization
    total_zone_trades = sum(r.num_trades for r in zone_results)
    zone_utilization = {}
    for z in ["normal", "cruise", "caution", "hail_mary"]:
        zt = sum(r.zone_trades.get(z, 0) for r in zone_results)
        zone_utilization[z] = zt / total_zone_trades if total_zone_trades > 0 else 0

    # Challenge statistics
    if zone_passed:
        avg_trades_pass = float(np.mean([r.num_trades for r in zone_passed]))
        avg_days_pass = float(np.mean([r.num_days for r in zone_passed]))
        avg_dd_pass = float(np.mean([r.max_drawdown for r in zone_passed]))
        avg_daily_dd_pass = float(np.mean([r.max_daily_drawdown for r in zone_passed]))
    else:
        avg_trades_pass = avg_days_pass = avg_dd_pass = avg_daily_dd_pass = 0

    stats = {
        "simulations": n,
        "strategy": {
            "win_rate": win_rate,
            "reward_risk": reward_risk,
            "per_trade_ev": win_rate * reward_risk - (1 - win_rate),
        },
        "challenge": {
            "pass_rate": zone_pass_rate,
            "pass_count": len(zone_passed),
            "avg_trades_to_pass": avg_trades_pass,
            "avg_days_to_pass": avg_days_pass,
            "avg_peak_dd_passed": avg_dd_pass,
            "avg_daily_dd_passed": avg_daily_dd_pass,
            "fail_overall_dd": fail_dd,
            "fail_daily_dd": fail_daily,
            "fail_time": max(0, fail_time),
            "zone_utilization": zone_utilization,
        },
        "funded": {
            "avg_payout": avg_payout,
            "median_payout": median_payout,
            "std_payout": payout_std,
            "payout_rate": payout_rate,
            "blow_rate": blow_rate,
            "avg_trades": avg_funded_trades,
            "percentiles": pctiles,
        },
        "combined": {
            "ev_per_attempt": ev_per_attempt,
            "challenge_fee": cfg.CHALLENGE_FEE,
            "net_ev": net_ev,
            "roi_per_attempt": net_ev / cfg.CHALLENGE_FEE
            if cfg.CHALLENGE_FEE > 0 else 0,
        },
        "flat_baseline": flat_stats,
    }

    if verbose:
        _print_report(stats)

    return stats


# ── Report printer ─────────────────────────────────────────────────

def _print_report(stats: dict):
    c = stats["challenge"]
    f = stats["funded"]
    x = stats["combined"]
    s = stats["strategy"]
    fb = stats.get("flat_baseline")

    print()
    print("=" * 68)
    print("  PROP FIRM CHALLENGE — ZONE-BASED SIZING SIMULATION")
    print("=" * 68)
    print()
    print(f"  Simulations:        {stats['simulations']:,}")
    print(f"  Account size:       ${cfg.ACCOUNT_SIZE:,.0f}")
    print(f"  Challenge fee:      ${cfg.CHALLENGE_FEE:,.0f}")
    print(f"  Profit target:      {cfg.PROFIT_TARGET_PCT:.0%}"
          f" (${cfg.ACCOUNT_SIZE * cfg.PROFIT_TARGET_PCT:,.0f})")
    print(f"  Max drawdown:       {cfg.MAX_DRAWDOWN_PCT:.0%}"
          f" (${cfg.ACCOUNT_SIZE * cfg.MAX_DRAWDOWN_PCT:,.0f})")
    print(f"  Daily drawdown:     {cfg.MAX_DAILY_DRAWDOWN_PCT:.0%}"
          f" (${cfg.ACCOUNT_SIZE * cfg.MAX_DAILY_DRAWDOWN_PCT:,.0f})")
    print(f"  Strategy:           WR={s['win_rate']:.1%}"
          f"  RR=1:{s['reward_risk']}"
          f"  EV/trade=${s['per_trade_ev']:.4f}")
    print()

    # ── Zone sizing config ──
    z = cfg.ZONE_SIZING
    print("─" * 68)
    print("  ZONE CONFIGURATION")
    print("─" * 68)
    print(f"  Normal:    {z['risk_normal']:.1%} risk"
          f"   (equity >= initial, < cruise)")
    print(f"  Cruise:    {z['risk_cruise']:.1%} risk"
          f"   (>= {z['cruise_threshold']:.0%} of target reached)")
    print(f"  Caution:   {z['risk_caution']:.1%} risk"
          f"   (equity < initial)")
    print(f"  Hail Mary: {z['risk_hail_mary']:.1%} risk"
          f"   (<= {z['hail_mary_threshold']:.0%} DD buffer left)")
    print(f"  Daily stop: {z['daily_stop_pct']:.0%} of daily DD used")
    print()

    # ── Challenge results ──
    print("─" * 68)
    print("  PHASE 1: Challenge (zone-based sizing)")
    print("─" * 68)
    print(f"  Pass rate:          {c['pass_rate']:.2%}"
          f"  ({c['pass_count']:,} / {stats['simulations']:,})")
    print(f"  Avg trades to pass: {c['avg_trades_to_pass']:.0f}")
    print(f"  Avg days to pass:   {c['avg_days_to_pass']:.1f}")
    print(f"  Avg peak DD (pass): {c['avg_peak_dd_passed']:.2%}")
    print(f"  Avg daily DD (pass):{c['avg_daily_dd_passed']:.2%}")
    print()
    print(f"  Failure breakdown:")
    total_fails = stats['simulations'] - c['pass_count']
    if total_fails > 0:
        print(f"    Overall DD:       {c['fail_overall_dd']:>6,}"
              f"  ({c['fail_overall_dd']/total_fails:.0%})")
        print(f"    Daily DD:         {c['fail_daily_dd']:>6,}"
              f"  ({c['fail_daily_dd']/total_fails:.0%})")
        print(f"    Time expired:     {c['fail_time']:>6,}"
              f"  ({c['fail_time']/total_fails:.0%})")
    print()
    print(f"  Zone utilization (% of total trades):")
    for zone_name, pct in c['zone_utilization'].items():
        print(f"    {zone_name:>12}: {pct:.1%}")
    print()

    # ── Flat baseline comparison ──
    if fb:
        print("─" * 68)
        print("  BASELINE: Flat 2% sizing (no zones)")
        print("─" * 68)
        print(f"  Pass rate:          {fb['pass_rate']:.2%}"
              f"  ({fb['pass_count']:,} / {stats['simulations']:,})")
        print(f"  Avg payout:         ${fb['avg_payout']:,.0f}")
        print(f"  Net EV:             ${fb['net_ev']:,.0f}")
        print(f"  ROI:                {fb['roi']:.1%}")
        total_fails_f = stats['simulations'] - fb['pass_count']
        if total_fails_f > 0:
            print(f"  Failures: DD={fb['fail_overall_dd']:,}"
                  f"  DailyDD={fb['fail_daily_dd']:,}"
                  f"  Time={fb['fail_time']:,}")
        print()

        # Delta
        delta_pass = c['pass_rate'] - fb['pass_rate']
        delta_ev = x['net_ev'] - fb['net_ev']
        print(f"  ZONE vs FLAT:")
        print(f"    Pass rate delta:  {delta_pass:+.2%}")
        print(f"    Net EV delta:     ${delta_ev:+,.0f}")
        print()

    # ── Funded results ──
    print("─" * 68)
    print("  PHASE 2: Funded (dd_frac)")
    print("─" * 68)
    print(f"  DD fraction:        {cfg.DD_FRAC_40PCT['dd_fraction']:.0%}"
          f" of remaining buffer")
    print(f"  Payout split:       {cfg.FUNDED_PAYOUT_SPLIT:.0%}")
    print(f"  Payout rate:        {f['payout_rate']:.1%}")
    print(f"  Blow-up rate:       {f['blow_rate']:.1%}")
    print(f"  Avg trades:         {f['avg_trades']:.0f}")
    print()
    print(f"  Avg payout:         ${f['avg_payout']:,.0f}")
    print(f"  Median payout:      ${f['median_payout']:,.0f}")
    print(f"  Std dev:            ${f['std_payout']:,.0f}")
    print()
    p = f["percentiles"]
    print(f"  Payout distribution:")
    for k, v in p.items():
        print(f"    {k.upper():>4}:  ${v:>10,.0f}")
    print()

    # ── Combined EV ──
    print("─" * 68)
    print("  COMBINED EXPECTED VALUE")
    print("─" * 68)
    print(f"  EV per attempt:     ${x['ev_per_attempt']:,.0f}"
          f"  (P(pass) x E[payout])")
    print(f"  Challenge fee:     -${x['challenge_fee']:,.0f}")
    print(f"  {'─' * 30}")
    print(f"  Net EV:             ${x['net_ev']:,.0f}")
    print(f"  ROI per attempt:    {x['roi_per_attempt']:.1%}")
    print()

    if x['net_ev'] > 0:
        breakeven = cfg.CHALLENGE_FEE / x['net_ev'] if x['net_ev'] > 0 else float('inf')
        print(f"  >> POSITIVE EDGE DETECTED")
        print(f"     Zero-EV trading → +EV via convex payoff + zone sizing.")
        print(f"     Expected breakeven after {breakeven:.1f} attempts.")
    else:
        print(f"  >> Negative EV — adjust zone parameters or strategy stats.")
    print()
    print("=" * 68)


if __name__ == "__main__":
    run_monte_carlo()
