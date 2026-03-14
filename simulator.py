"""
Monte Carlo simulator for prop firm challenge strategies.

Two-phase approach:
  Phase 1 (frac_2pct):  Pass the challenge using zero-EV ORB with
                         fixed-fractional 2% sizing. Convex payoff
                         structure yields ~41% pass rate.

  Phase 2 (dd_frac_40pct): Once funded, maximize payout by risking
                            40% of remaining drawdown buffer per trade.
                            Extract maximum value before account blows.

Combined EV = P(pass) * E[funded_payout] - (1 - P(pass)) * 0
            = P(pass) * E[funded_payout]
Net EV      = Combined EV - challenge_fee

If Net EV > 0, the strategy has positive expected value despite
each individual trade having zero edge.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sim_config as cfg


@dataclass
class TradeResult:
    pnl: float
    equity_after: float
    drawdown: float
    risk_amount: float
    won: bool


@dataclass
class ChallengeResult:
    passed: bool
    final_equity: float
    peak_equity: float
    max_drawdown: float
    num_trades: int
    num_days: int
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class FundedResult:
    final_equity: float
    peak_equity: float
    max_drawdown: float
    gross_profit: float
    payout: float          # After split
    num_trades: int
    blown: bool            # Hit max drawdown
    equity_curve: list[float] = field(default_factory=list)


def simulate_challenge(rng: np.random.Generator,
                        strategy: dict = None,
                        record_curve: bool = False) -> ChallengeResult:
    """
    Simulate a single challenge attempt using frac_2pct.

    Rules:
      - Start at ACCOUNT_SIZE
      - Hit PROFIT_TARGET_PCT to pass
      - Hit MAX_DRAWDOWN_PCT to fail
      - Must trade MIN_TRADING_DAYS before passing
      - Must finish within MAX_TRADING_DAYS
    """
    if strategy is None:
        strategy = cfg.FRAC_2PCT

    equity = cfg.ACCOUNT_SIZE
    initial = cfg.ACCOUNT_SIZE
    peak = equity
    profit_target = initial * (1 + cfg.PROFIT_TARGET_PCT)
    max_dd_level = initial * (1 - cfg.MAX_DRAWDOWN_PCT)

    trades = 0
    curve = [equity] if record_curve else []

    win_rate = strategy["win_rate"]
    rr = strategy["reward_risk_ratio"]
    risk_pct = strategy["risk_per_trade"]

    total_trades = cfg.MAX_TRADING_DAYS * cfg.TRADES_PER_DAY

    for day in range(cfg.MAX_TRADING_DAYS):
        day_start_equity = equity

        for _ in range(cfg.TRADES_PER_DAY):
            # Position sizing: fixed fractional
            risk_amount = equity * risk_pct

            # Simulate trade outcome
            if rng.random() < win_rate:
                pnl = risk_amount * rr  # Win
            else:
                pnl = -risk_amount       # Loss

            equity += pnl
            trades += 1
            peak = max(peak, equity)

            if record_curve:
                curve.append(equity)

            # Check daily drawdown
            daily_dd = (day_start_equity - equity) / initial
            if daily_dd >= cfg.MAX_DAILY_DRAWDOWN_PCT:
                max_dd = (peak - equity) / initial
                return ChallengeResult(
                    passed=False, final_equity=equity, peak_equity=peak,
                    max_drawdown=max_dd, num_trades=trades,
                    num_days=day + 1, equity_curve=curve)

            # Check max drawdown (from initial balance)
            if equity <= max_dd_level:
                max_dd = (initial - equity) / initial
                return ChallengeResult(
                    passed=False, final_equity=equity, peak_equity=peak,
                    max_drawdown=max_dd, num_trades=trades,
                    num_days=day + 1, equity_curve=curve)

            # Check profit target (only after min trading days)
            if (day + 1) >= cfg.MIN_TRADING_DAYS and equity >= profit_target:
                max_dd = (peak - equity) / initial if peak > equity else 0
                return ChallengeResult(
                    passed=True, final_equity=equity, peak_equity=peak,
                    max_drawdown=max_dd, num_trades=trades,
                    num_days=day + 1, equity_curve=curve)

    # Time expired without passing
    max_dd = (peak - equity) / initial if peak > equity else 0
    return ChallengeResult(
        passed=False, final_equity=equity, peak_equity=peak,
        max_drawdown=max_dd, num_trades=trades,
        num_days=cfg.MAX_TRADING_DAYS, equity_curve=curve)


def simulate_funded(rng: np.random.Generator,
                     strategy: dict = None,
                     record_curve: bool = False) -> FundedResult:
    """
    Simulate funded account using dd_frac_40pct.

    Sizing: risk = dd_fraction * remaining_buffer
    where remaining_buffer = max_dd_limit - current_drawdown_from_peak

    The aggressive sizing extracts maximum payout before the
    inevitable blowup. This is the optimal strategy because:
      - You already "won" (passed challenge)
      - The funded account is a free option
      - Maximize the option's payoff by being maximally aggressive
    """
    if strategy is None:
        strategy = cfg.DD_FRAC_40PCT

    equity = cfg.ACCOUNT_SIZE
    initial = cfg.ACCOUNT_SIZE
    peak = equity
    max_dd_level = initial * (1 - cfg.FUNDED_MAX_DRAWDOWN_PCT)
    max_dd_dollars = initial * cfg.FUNDED_MAX_DRAWDOWN_PCT

    win_rate = strategy["win_rate"]
    rr = strategy["reward_risk_ratio"]
    dd_frac = strategy["dd_fraction"]

    trades = 0
    high_water = equity
    curve = [equity] if record_curve else []

    for _ in range(cfg.FUNDED_MAX_TRADES):
        # Remaining drawdown buffer
        current_dd = max(0, high_water - equity)
        remaining_buffer = max_dd_dollars - current_dd

        if remaining_buffer <= 0:
            break

        # Position sizing: fraction of remaining buffer
        risk_amount = remaining_buffer * dd_frac

        # Floor: don't risk less than $100 (not worth the trade)
        if risk_amount < 100:
            break

        if rng.random() < win_rate:
            pnl = risk_amount * rr
        else:
            pnl = -risk_amount

        equity += pnl
        trades += 1
        high_water = max(high_water, equity)
        peak = max(peak, equity)

        if record_curve:
            curve.append(equity)

        # Check if blown
        if equity <= max_dd_level:
            gross = max(0, peak - initial)
            return FundedResult(
                final_equity=equity, peak_equity=peak,
                max_drawdown=(initial - equity) / initial,
                gross_profit=gross,
                payout=gross * cfg.FUNDED_PAYOUT_SPLIT,
                num_trades=trades, blown=True,
                equity_curve=curve)

    # Survived all trades or buffer exhausted — take payout
    gross = max(0, equity - initial)
    return FundedResult(
        final_equity=equity, peak_equity=peak,
        max_drawdown=(peak - equity) / initial if peak > equity else 0,
        gross_profit=gross,
        payout=gross * cfg.FUNDED_PAYOUT_SPLIT,
        num_trades=trades, blown=False,
        equity_curve=curve)


def run_monte_carlo(n: int = None, seed: int = None,
                     verbose: bool = True) -> dict:
    """
    Run full two-phase Monte Carlo simulation.

    Returns statistics on:
      - Challenge pass rate
      - Funded payout distribution
      - Combined EV per attempt
      - Net EV after challenge fee
      - Kelly-optimal number of concurrent challenges
    """
    if n is None:
        n = cfg.MC_SIMULATIONS
    if seed is None:
        seed = cfg.MC_SEED

    rng = np.random.default_rng(seed)

    # Phase 1: Challenge attempts
    challenge_results = []
    for _ in range(n):
        result = simulate_challenge(rng)
        challenge_results.append(result)

    passed = [r for r in challenge_results if r.passed]
    pass_rate = len(passed) / n

    # Phase 2: Funded accounts (only for those who passed)
    funded_results = []
    funded_payouts = []
    for _ in range(len(passed)):
        result = simulate_funded(rng)
        funded_results.append(result)
        funded_payouts.append(result.payout)

    funded_payouts = np.array(funded_payouts) if funded_payouts else np.array([0.0])

    # Combined statistics
    avg_funded_payout = np.mean(funded_payouts) if len(funded_payouts) > 0 else 0
    median_funded_payout = np.median(funded_payouts) if len(funded_payouts) > 0 else 0

    # EV per challenge attempt
    ev_per_attempt = pass_rate * avg_funded_payout
    net_ev = ev_per_attempt - cfg.CHALLENGE_FEE

    # ROI per attempt
    roi = net_ev / cfg.CHALLENGE_FEE if cfg.CHALLENGE_FEE > 0 else float('inf')

    # Payout percentiles (conditional on passing)
    if len(funded_payouts) > 0:
        p10 = np.percentile(funded_payouts, 10)
        p25 = np.percentile(funded_payouts, 25)
        p50 = np.percentile(funded_payouts, 50)
        p75 = np.percentile(funded_payouts, 75)
        p90 = np.percentile(funded_payouts, 90)
        p99 = np.percentile(funded_payouts, 99)
        payout_std = np.std(funded_payouts)
    else:
        p10 = p25 = p50 = p75 = p90 = p99 = payout_std = 0

    # Fraction of funded accounts that actually produce a payout
    if len(funded_results) > 0:
        payout_rate = sum(1 for r in funded_results if r.payout > 0) / len(funded_results)
        blow_rate = sum(1 for r in funded_results if r.blown) / len(funded_results)
    else:
        payout_rate = blow_rate = 0

    stats = {
        "simulations": n,
        "challenge": {
            "pass_rate": pass_rate,
            "pass_count": len(passed),
            "avg_trades_to_pass": np.mean([r.num_trades for r in passed]) if passed else 0,
            "avg_days_to_pass": np.mean([r.num_days for r in passed]) if passed else 0,
            "avg_peak_dd_passed": np.mean([r.max_drawdown for r in passed]) if passed else 0,
        },
        "funded": {
            "avg_payout": avg_funded_payout,
            "median_payout": median_funded_payout,
            "std_payout": payout_std,
            "payout_rate": payout_rate,
            "blow_rate": blow_rate,
            "avg_trades": np.mean([r.num_trades for r in funded_results]) if funded_results else 0,
            "percentiles": {
                "p10": p10, "p25": p25, "p50": p50,
                "p75": p75, "p90": p90, "p99": p99,
            },
        },
        "combined": {
            "ev_per_attempt": ev_per_attempt,
            "challenge_fee": cfg.CHALLENGE_FEE,
            "net_ev": net_ev,
            "roi_per_attempt": roi,
        },
    }

    if verbose:
        _print_report(stats)

    return stats


def _print_report(stats: dict):
    """Pretty-print simulation results."""
    c = stats["challenge"]
    f = stats["funded"]
    x = stats["combined"]

    print("=" * 64)
    print("  PROP FIRM CHALLENGE — ZERO-EV STRATEGY SIMULATION")
    print("=" * 64)
    print()
    print(f"  Simulations:        {stats['simulations']:,}")
    print(f"  Account size:       ${cfg.ACCOUNT_SIZE:,.0f}")
    print(f"  Challenge fee:      ${cfg.CHALLENGE_FEE:,.0f}")
    print(f"  Profit target:      {cfg.PROFIT_TARGET_PCT:.0%}")
    print(f"  Max drawdown:       {cfg.MAX_DRAWDOWN_PCT:.0%}")
    print()

    print("─" * 64)
    print("  PHASE 1: frac_2pct (Challenge)")
    print("─" * 64)
    print(f"  Strategy:           ORB · WR={cfg.FRAC_2PCT['win_rate']:.1%}"
          f" · RR=1:{cfg.FRAC_2PCT['reward_risk_ratio']}"
          f" · Risk={cfg.FRAC_2PCT['risk_per_trade']:.0%}/trade")
    print(f"  Per-trade EV:       ${0:.2f} (zero)")
    print(f"  Pass rate:          {c['pass_rate']:.1%}"
          f"  ({c['pass_count']:,} / {stats['simulations']:,})")
    print(f"  Avg trades to pass: {c['avg_trades_to_pass']:.0f}")
    print(f"  Avg days to pass:   {c['avg_days_to_pass']:.1f}")
    print(f"  Avg peak DD (pass): {c['avg_peak_dd_passed']:.2%}")
    print()

    print("─" * 64)
    print("  PHASE 2: dd_frac_40pct (Funded)")
    print("─" * 64)
    print(f"  Strategy:           ORB · WR={cfg.DD_FRAC_40PCT['win_rate']:.1%}"
          f" · RR=1:{cfg.DD_FRAC_40PCT['reward_risk_ratio']}"
          f" · Risk={cfg.DD_FRAC_40PCT['dd_fraction']:.0%} of DD buffer")
    print(f"  Payout split:       {cfg.FUNDED_PAYOUT_SPLIT:.0%} to trader")
    print(f"  Payout rate:        {f['payout_rate']:.1%} of funded accounts")
    print(f"  Blow-up rate:       {f['blow_rate']:.1%}")
    print(f"  Avg trades:         {f['avg_trades']:.0f}")
    print()
    print(f"  Avg payout:         ${f['avg_payout']:,.0f}")
    print(f"  Median payout:      ${f['median_payout']:,.0f}")
    print(f"  Std dev:            ${f['std_payout']:,.0f}")
    print()
    print(f"  Payout distribution (conditional on passing):")
    p = f["percentiles"]
    print(f"    P10:  ${p['p10']:>10,.0f}")
    print(f"    P25:  ${p['p25']:>10,.0f}")
    print(f"    P50:  ${p['p50']:>10,.0f}")
    print(f"    P75:  ${p['p75']:>10,.0f}")
    print(f"    P90:  ${p['p90']:>10,.0f}")
    print(f"    P99:  ${p['p99']:>10,.0f}")
    print()

    print("─" * 64)
    print("  COMBINED EXPECTED VALUE")
    print("─" * 64)
    print(f"  EV per attempt:     ${x['ev_per_attempt']:,.0f}"
          f"  (P(pass) × E[payout])")
    print(f"  Challenge fee:     -${x['challenge_fee']:,.0f}")
    print(f"  ────────────────────────────")
    print(f"  Net EV:             ${x['net_ev']:,.0f}")
    print(f"  ROI per attempt:    {x['roi_per_attempt']:.1%}")
    print()

    if x['net_ev'] > 0:
        attempts_to_recover = cfg.CHALLENGE_FEE / x['net_ev'] if x['net_ev'] > 0 else float('inf')
        print(f"  ✓ POSITIVE EDGE DETECTED")
        print(f"    Zero-EV trading becomes +EV via convex payoff structure.")
        print(f"    Expected breakeven after {attempts_to_recover:.1f} attempts.")
    else:
        print(f"  ✗ Negative EV — adjust parameters.")

    print()
    print("=" * 64)


if __name__ == "__main__":
    run_monte_carlo()
