"""
Sensitivity analysis for prop firm challenge strategies.

Sweeps key parameters to find optimal configurations:
  - Risk per trade (frac_2pct phase)
  - DD fraction (dd_frac_40pct phase)
  - Win rate / RR combinations (all zero-EV)
  - Challenge fee vs account size
"""

import numpy as np
from copy import deepcopy
import sim_config as cfg
from simulator import simulate_challenge, simulate_funded


def sweep_risk_per_trade(risks=None, n=20_000, seed=42):
    """
    Sweep risk_per_trade for the challenge phase.
    Find the sweet spot between pass rate and blow-up rate.
    """
    if risks is None:
        risks = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("  SWEEP: Risk Per Trade (Challenge Phase)")
    print("=" * 70)
    print(f"  {'Risk%':>6}  {'Pass%':>6}  {'AvgTrades':>10}  {'AvgDays':>8}"
          f"  {'AvgDD%':>7}  {'NetEV':>8}")
    print("─" * 70)

    best_ev = -float('inf')
    best_risk = 0

    for risk in risks:
        strat = deepcopy(cfg.FRAC_2PCT)
        strat["risk_per_trade"] = risk

        sub_rng = np.random.default_rng(rng.integers(0, 2**32))

        passed = 0
        total_trades_passed = 0
        total_days_passed = 0
        total_dd_passed = 0
        payouts = []

        for _ in range(n):
            cr = simulate_challenge(sub_rng, strategy=strat)
            if cr.passed:
                passed += 1
                total_trades_passed += cr.num_trades
                total_days_passed += cr.num_days
                total_dd_passed += cr.max_drawdown

                fr = simulate_funded(sub_rng)
                payouts.append(fr.payout)

        pass_rate = passed / n
        avg_trades = total_trades_passed / passed if passed else 0
        avg_days = total_days_passed / passed if passed else 0
        avg_dd = total_dd_passed / passed if passed else 0
        avg_payout = np.mean(payouts) if payouts else 0
        net_ev = pass_rate * avg_payout - cfg.CHALLENGE_FEE

        if net_ev > best_ev:
            best_ev = net_ev
            best_risk = risk

        print(f"  {risk:>5.1%}  {pass_rate:>5.1%}  {avg_trades:>10.0f}"
              f"  {avg_days:>7.1f}  {avg_dd:>6.2%}  ${net_ev:>7,.0f}")

    print("─" * 70)
    print(f"  Best: {best_risk:.1%} risk → Net EV = ${best_ev:,.0f}")
    print()


def sweep_dd_fraction(fractions=None, n=20_000, seed=42):
    """
    Sweep dd_fraction for the funded phase.
    Find the optimal aggression level.
    """
    if fractions is None:
        fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    rng = np.random.default_rng(seed)

    # First, generate a fixed set of challenge passes
    sub_rng = np.random.default_rng(rng.integers(0, 2**32))

    print("=" * 70)
    print("  SWEEP: DD Fraction (Funded Phase)")
    print("=" * 70)
    print(f"  {'DDFrac%':>7}  {'AvgPayout':>10}  {'MedPayout':>10}"
          f"  {'P90Payout':>10}  {'PayoutRate':>10}")
    print("─" * 70)

    best_avg = 0
    best_frac = 0

    for frac in fractions:
        strat = deepcopy(cfg.DD_FRAC_40PCT)
        strat["dd_fraction"] = frac

        sub_rng2 = np.random.default_rng(rng.integers(0, 2**32))
        payouts = []
        payout_count = 0

        for _ in range(n):
            fr = simulate_funded(sub_rng2, strategy=strat)
            payouts.append(fr.payout)
            if fr.payout > 0:
                payout_count += 1

        payouts = np.array(payouts)
        avg = np.mean(payouts)
        med = np.median(payouts)
        p90 = np.percentile(payouts, 90)
        pr = payout_count / n

        if avg > best_avg:
            best_avg = avg
            best_frac = frac

        print(f"  {frac:>6.0%}  ${avg:>9,.0f}  ${med:>9,.0f}"
              f"  ${p90:>9,.0f}  {pr:>9.1%}")

    print("─" * 70)
    print(f"  Best: {best_frac:.0%} DD frac → Avg payout = ${best_avg:,.0f}")
    print()


def sweep_zero_ev_params(n=20_000, seed=42):
    """
    Sweep different zero-EV (win_rate, RR) combinations.
    All have EV=0 but different variance profiles.

    Zero-EV constraint: WR * RR = (1 - WR) * 1
    → RR = (1 - WR) / WR
    """
    win_rates = [0.50, 0.55, 0.60, 2/3, 0.70, 0.75, 0.80, 0.85, 0.90]

    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("  SWEEP: Zero-EV (WinRate, RR) Combinations")
    print("  Constraint: WR × RR = (1-WR) × 1  →  EV = 0")
    print("=" * 70)
    print(f"  {'WR%':>5}  {'RR':>6}  {'Pass%':>6}  {'AvgPayout':>10}"
          f"  {'NetEV':>8}  {'ROI':>7}")
    print("─" * 70)

    best_ev = -float('inf')
    best_wr = 0

    for wr in win_rates:
        rr = (1 - wr) / wr  # Zero-EV constraint

        strat_c = deepcopy(cfg.FRAC_2PCT)
        strat_c["win_rate"] = wr
        strat_c["reward_risk_ratio"] = rr

        strat_f = deepcopy(cfg.DD_FRAC_40PCT)
        strat_f["win_rate"] = wr
        strat_f["reward_risk_ratio"] = rr

        sub_rng = np.random.default_rng(rng.integers(0, 2**32))

        passed = 0
        payouts = []

        for _ in range(n):
            cr = simulate_challenge(sub_rng, strategy=strat_c)
            if cr.passed:
                passed += 1
                fr = simulate_funded(sub_rng, strategy=strat_f)
                payouts.append(fr.payout)

        pass_rate = passed / n
        avg_payout = np.mean(payouts) if payouts else 0
        net_ev = pass_rate * avg_payout - cfg.CHALLENGE_FEE
        roi = net_ev / cfg.CHALLENGE_FEE

        if net_ev > best_ev:
            best_ev = net_ev
            best_wr = wr

        print(f"  {wr:>4.0%}  {rr:>5.2f}  {pass_rate:>5.1%}"
              f"  ${avg_payout:>9,.0f}  ${net_ev:>7,.0f}  {roi:>6.0%}")

    print("─" * 70)
    rr_best = (1 - best_wr) / best_wr
    print(f"  Best: WR={best_wr:.0%} RR={rr_best:.2f} → Net EV = ${best_ev:,.0f}")
    print()


def multi_attempt_analysis(n_attempts=10, n_sims=20_000, seed=42):
    """
    Analyze the strategy across multiple sequential challenge attempts.
    Shows expected cumulative PnL over N attempts.
    """
    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("  MULTI-ATTEMPT ANALYSIS")
    print(f"  Simulating {n_sims:,} paths of {n_attempts} sequential attempts")
    print("=" * 70)

    cumulative_pnls = np.zeros((n_sims, n_attempts))

    for sim in range(n_sims):
        cumulative = 0
        for attempt in range(n_attempts):
            cumulative -= cfg.CHALLENGE_FEE  # Pay fee

            cr = simulate_challenge(rng)
            if cr.passed:
                fr = simulate_funded(rng)
                cumulative += fr.payout

            cumulative_pnls[sim, attempt] = cumulative

    print(f"\n  {'Attempt':>7}  {'AvgPnL':>10}  {'MedPnL':>10}"
          f"  {'P(profit)':>10}  {'P10':>10}  {'P90':>10}")
    print("─" * 70)

    for i in range(n_attempts):
        col = cumulative_pnls[:, i]
        avg = np.mean(col)
        med = np.median(col)
        p_profit = np.mean(col > 0)
        p10 = np.percentile(col, 10)
        p90 = np.percentile(col, 90)

        print(f"  {i+1:>7}  ${avg:>9,.0f}  ${med:>9,.0f}"
              f"  {p_profit:>9.1%}  ${p10:>9,.0f}  ${p90:>9,.0f}")

    print()


if __name__ == "__main__":
    sweep_risk_per_trade()
    sweep_dd_fraction()
    sweep_zero_ev_params()
    multi_attempt_analysis()
