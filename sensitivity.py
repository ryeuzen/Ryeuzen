"""
Sensitivity analysis for zone-based prop firm challenge strategies.

Sweeps key parameters to find optimal zone configurations:
  - Zone risk fractions (normal, cruise, caution, hail_mary)
  - Zone boundary thresholds (cruise, hail_mary)
  - Daily stop percentage
  - Win rate / RR combinations (all zero-EV)
  - Multi-attempt cumulative EV
"""

import numpy as np
from copy import deepcopy
import sim_config as cfg
from simulator import (simulate_challenge, simulate_challenge_flat,
                       simulate_funded, classify_zone)


def _run_challenge_batch(rng, n, *, win_rate=None, reward_risk=None,
                         zones=None, flat=False, risk_pct=None):
    """Run N challenges and return (pass_rate, avg_payout, net_ev, details)."""
    passed = 0
    payouts = []
    fail_dd = fail_daily = fail_time = 0

    for _ in range(n):
        if flat:
            cr = simulate_challenge_flat(rng, win_rate=win_rate,
                                         reward_risk=reward_risk,
                                         risk_pct=risk_pct)
        else:
            cr = simulate_challenge(rng, win_rate=win_rate,
                                    reward_risk=reward_risk, zones=zones)
        if cr.passed:
            passed += 1
            fr = simulate_funded(rng, win_rate=win_rate,
                                 reward_risk=reward_risk)
            payouts.append(fr.payout)
        else:
            if cr.max_drawdown >= cfg.MAX_DRAWDOWN_PCT - 0.001:
                fail_dd += 1
            elif cr.max_daily_drawdown >= cfg.MAX_DAILY_DRAWDOWN_PCT - 0.001:
                fail_daily += 1
            else:
                fail_time += 1

    pass_rate = passed / n
    avg_payout = float(np.mean(payouts)) if payouts else 0
    net_ev = pass_rate * avg_payout - cfg.CHALLENGE_FEE

    return pass_rate, avg_payout, net_ev, {
        "fail_dd": fail_dd, "fail_daily": fail_daily, "fail_time": fail_time}


# ── Sweep: zone risk fractions ─────────────────────────────────────

def sweep_zone_risks(n=20_000, seed=42):
    """
    Sweep risk fractions for each zone independently while holding
    others constant. Find which zone's risk has the most impact.
    """
    print("=" * 74)
    print("  SWEEP: Zone Risk Fractions")
    print("=" * 74)

    base_zones = deepcopy(cfg.ZONE_SIZING)

    for zone_name, risk_key in [
        ("normal", "risk_normal"),
        ("cruise", "risk_cruise"),
        ("caution", "risk_caution"),
        ("hail_mary", "risk_hail_mary"),
    ]:
        risks = {
            "normal": [0.010, 0.015, 0.020, 0.025, 0.030, 0.035],
            "cruise": [0.005, 0.008, 0.010, 0.015, 0.020],
            "caution": [0.005, 0.008, 0.010, 0.012, 0.015, 0.020],
            "hail_mary": [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15],
        }[zone_name]

        print(f"\n  Zone: {zone_name} (others held at defaults)")
        print(f"  {'Risk%':>6}  {'Pass%':>6}  {'AvgPay':>8}  {'NetEV':>8}")
        print("  " + "─" * 40)

        best_ev = -float('inf')
        best_risk = 0

        for risk in risks:
            zones = deepcopy(base_zones)
            zones[risk_key] = risk

            rng = np.random.default_rng(seed)
            pr, ap, nev, _ = _run_challenge_batch(
                rng, n, zones=zones)

            if nev > best_ev:
                best_ev = nev
                best_risk = risk

            print(f"  {risk:>5.1%}  {pr:>5.1%}  ${ap:>7,.0f}  ${nev:>7,.0f}")

        print(f"  Best: {best_risk:.1%} → Net EV = ${best_ev:,.0f}")


# ── Sweep: zone boundary thresholds ────────────────────────────────

def sweep_thresholds(n=20_000, seed=42):
    """
    Sweep cruise_threshold and hail_mary_threshold.
    """
    print("\n" + "=" * 74)
    print("  SWEEP: Zone Boundary Thresholds")
    print("=" * 74)

    # Cruise threshold
    print(f"\n  Cruise threshold (% of target reached before conservative sizing)")
    print(f"  {'Thresh':>7}  {'Pass%':>6}  {'AvgPay':>8}  {'NetEV':>8}")
    print("  " + "─" * 40)

    best_ev = -float('inf')
    best_t = 0

    for thresh in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        zones = deepcopy(cfg.ZONE_SIZING)
        zones["cruise_threshold"] = thresh

        rng = np.random.default_rng(seed)
        pr, ap, nev, _ = _run_challenge_batch(rng, n, zones=zones)

        if nev > best_ev:
            best_ev = nev
            best_t = thresh

        print(f"  {thresh:>6.0%}  {pr:>5.1%}  ${ap:>7,.0f}  ${nev:>7,.0f}")

    print(f"  Best: {best_t:.0%} → Net EV = ${best_ev:,.0f}")

    # Hail Mary threshold
    print(f"\n  Hail Mary threshold (% DD buffer remaining to trigger)")
    print(f"  {'Thresh':>7}  {'Pass%':>6}  {'AvgPay':>8}  {'NetEV':>8}")
    print("  " + "─" * 40)

    best_ev = -float('inf')
    best_t = 0

    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        zones = deepcopy(cfg.ZONE_SIZING)
        zones["hail_mary_threshold"] = thresh

        rng = np.random.default_rng(seed)
        pr, ap, nev, _ = _run_challenge_batch(rng, n, zones=zones)

        if nev > best_ev:
            best_ev = nev
            best_t = thresh

        print(f"  {thresh:>6.0%}  {pr:>5.1%}  ${ap:>7,.0f}  ${nev:>7,.0f}")

    print(f"  Best: {best_t:.0%} → Net EV = ${best_ev:,.0f}")


# ── Sweep: daily stop percentage ───────────────────────────────────

def sweep_daily_stop(n=20_000, seed=42):
    """
    Sweep daily_stop_pct: fraction of daily DD used before halting.
    Lower = more conservative (stop earlier), higher = more trades.
    """
    print("\n" + "=" * 74)
    print("  SWEEP: Daily Stop Percentage")
    print("=" * 74)
    print(f"  {'Stop%':>6}  {'Pass%':>6}  {'AvgPay':>8}  {'NetEV':>8}"
          f"  {'FailDD':>7}  {'FailDay':>7}")
    print("  " + "─" * 55)

    best_ev = -float('inf')
    best_pct = 0

    for pct in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        zones = deepcopy(cfg.ZONE_SIZING)
        zones["daily_stop_pct"] = pct

        rng = np.random.default_rng(seed)
        pr, ap, nev, details = _run_challenge_batch(rng, n, zones=zones)

        if nev > best_ev:
            best_ev = nev
            best_pct = pct

        print(f"  {pct:>5.0%}  {pr:>5.1%}  ${ap:>7,.0f}  ${nev:>7,.0f}"
              f"  {details['fail_dd']:>7,}  {details['fail_daily']:>7,}")

    print(f"  Best: {best_pct:.0%} → Net EV = ${best_ev:,.0f}")


# ── Sweep: zero-EV strategy parameters ────────────────────────────

def sweep_zero_ev_params(n=20_000, seed=42):
    """
    Sweep different zero-EV (win_rate, RR) combinations with zones.
    All have EV=0 but different variance profiles.

    Zero-EV constraint: WR * RR = (1 - WR) * 1.0
    → RR = (1 - WR) / WR

    Higher WR + lower RR = lower variance (more frequent small wins)
    Lower WR + higher RR = higher variance (less frequent big wins)
    """
    win_rates = [0.50, 0.55, 0.60, 2/3, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("\n" + "=" * 74)
    print("  SWEEP: Zero-EV (WR, RR) Combinations — Zone-Based")
    print("  Constraint: WR * RR = (1-WR)  →  EV/trade = 0")
    print("=" * 74)
    print(f"  {'WR%':>5}  {'RR':>6}  {'Pass%':>6}  {'AvgPay':>10}"
          f"  {'NetEV':>8}  {'ROI':>7}")
    print("  " + "─" * 55)

    best_ev = -float('inf')
    best_wr = 0

    for wr in win_rates:
        rr = (1 - wr) / wr

        rng = np.random.default_rng(seed)
        pr, ap, nev, _ = _run_challenge_batch(
            rng, n, win_rate=wr, reward_risk=rr)

        roi = nev / cfg.CHALLENGE_FEE if cfg.CHALLENGE_FEE > 0 else 0

        if nev > best_ev:
            best_ev = nev
            best_wr = wr

        print(f"  {wr:>4.0%}  {rr:>5.2f}  {pr:>5.1%}"
              f"  ${ap:>9,.0f}  ${nev:>7,.0f}  {roi:>6.0%}")

    rr_best = (1 - best_wr) / best_wr
    print(f"  Best: WR={best_wr:.0%} RR={rr_best:.2f}"
          f" → Net EV = ${best_ev:,.0f}")


# ── Sweep: flat risk (baseline comparison) ─────────────────────────

def sweep_flat_risk(n=20_000, seed=42):
    """Sweep flat risk percentage for baseline comparison."""
    risks = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

    print("\n" + "=" * 74)
    print("  SWEEP: Flat Risk Per Trade (Baseline — No Zones)")
    print("=" * 74)
    print(f"  {'Risk%':>6}  {'Pass%':>6}  {'AvgPay':>8}  {'NetEV':>8}"
          f"  {'FailDD':>7}  {'FailDay':>7}")
    print("  " + "─" * 55)

    best_ev = -float('inf')
    best_risk = 0

    for risk in risks:
        rng = np.random.default_rng(seed)
        pr, ap, nev, details = _run_challenge_batch(
            rng, n, flat=True, risk_pct=risk)

        if nev > best_ev:
            best_ev = nev
            best_risk = risk

        print(f"  {risk:>5.1%}  {pr:>5.1%}  ${ap:>7,.0f}  ${nev:>7,.0f}"
              f"  {details['fail_dd']:>7,}  {details['fail_daily']:>7,}")

    print(f"  Best: {best_risk:.1%} → Net EV = ${best_ev:,.0f}")


# ── Multi-attempt analysis ─────────────────────────────────────────

def multi_attempt_analysis(n_attempts=10, n_sims=20_000, seed=42):
    """
    Analyze the strategy across multiple sequential challenge attempts.
    Shows expected cumulative PnL over N attempts (zone vs flat).
    """
    rng_zone = np.random.default_rng(seed)
    rng_flat = np.random.default_rng(seed)

    print("\n" + "=" * 74)
    print("  MULTI-ATTEMPT ANALYSIS")
    print(f"  {n_sims:,} paths x {n_attempts} sequential attempts")
    print("=" * 74)

    zone_pnls = np.zeros((n_sims, n_attempts))
    flat_pnls = np.zeros((n_sims, n_attempts))

    for sim in range(n_sims):
        z_cum = 0.0
        f_cum = 0.0
        for attempt in range(n_attempts):
            z_cum -= cfg.CHALLENGE_FEE
            f_cum -= cfg.CHALLENGE_FEE

            cr_z = simulate_challenge(rng_zone)
            if cr_z.passed:
                fr = simulate_funded(rng_zone)
                z_cum += fr.payout

            cr_f = simulate_challenge_flat(rng_flat)
            if cr_f.passed:
                fr = simulate_funded(rng_flat)
                f_cum += fr.payout

            zone_pnls[sim, attempt] = z_cum
            flat_pnls[sim, attempt] = f_cum

    print(f"\n  {'':>7}  {'─── Zone-Based ───':>35}  {'─── Flat 2% ───':>35}")
    print(f"  {'#':>3}  {'AvgPnL':>10}  {'MedPnL':>10}  {'P(+)':>6}"
          f"  {'AvgPnL':>10}  {'MedPnL':>10}  {'P(+)':>6}")
    print("  " + "─" * 65)

    for i in range(n_attempts):
        z = zone_pnls[:, i]
        f = flat_pnls[:, i]
        print(f"  {i+1:>3}"
              f"  ${np.mean(z):>9,.0f}  ${np.median(z):>9,.0f}"
              f"  {np.mean(z > 0):>5.0%}"
              f"  ${np.mean(f):>9,.0f}  ${np.median(f):>9,.0f}"
              f"  {np.mean(f > 0):>5.0%}")

    print()


# ── Joint zone optimization ───────────────────────────────────────

def optimize_zones(n=10_000, seed=42):
    """
    Grid search over key zone parameters to find the optimal
    configuration. Searches over:
      - risk_hail_mary
      - hail_mary_threshold
      - daily_stop_pct
    (These 3 have the most impact on pass rate.)
    """
    hm_risks = [0.04, 0.06, 0.08, 0.10]
    hm_thresholds = [0.20, 0.25, 0.30, 0.35]
    daily_stops = [0.60, 0.70, 0.80]

    print("\n" + "=" * 74)
    print("  GRID SEARCH: Joint Zone Optimization")
    print(f"  {len(hm_risks) * len(hm_thresholds) * len(daily_stops)} combinations")
    print("=" * 74)
    print(f"  {'HM_Risk':>7}  {'HM_Thresh':>9}  {'DayStop':>7}"
          f"  {'Pass%':>6}  {'NetEV':>8}")
    print("  " + "─" * 48)

    best_ev = -float('inf')
    best_params = {}

    for hm_risk in hm_risks:
        for hm_thresh in hm_thresholds:
            for ds in daily_stops:
                zones = deepcopy(cfg.ZONE_SIZING)
                zones["risk_hail_mary"] = hm_risk
                zones["hail_mary_threshold"] = hm_thresh
                zones["daily_stop_pct"] = ds

                rng = np.random.default_rng(seed)
                pr, ap, nev, _ = _run_challenge_batch(
                    rng, n, zones=zones)

                if nev > best_ev:
                    best_ev = nev
                    best_params = {
                        "risk_hail_mary": hm_risk,
                        "hail_mary_threshold": hm_thresh,
                        "daily_stop_pct": ds,
                        "pass_rate": pr,
                    }

                print(f"  {hm_risk:>6.0%}  {hm_thresh:>8.0%}  {ds:>6.0%}"
                      f"  {pr:>5.1%}  ${nev:>7,.0f}")

    print("  " + "─" * 48)
    print(f"  BEST: HM_Risk={best_params['risk_hail_mary']:.0%}"
          f"  HM_Thresh={best_params['hail_mary_threshold']:.0%}"
          f"  DayStop={best_params['daily_stop_pct']:.0%}"
          f"  → Pass={best_params['pass_rate']:.1%}"
          f"  Net EV=${best_ev:,.0f}")
    print()

    return best_params


if __name__ == "__main__":
    sweep_zone_risks()
    sweep_thresholds()
    sweep_daily_stop()
    sweep_zero_ev_params()
    sweep_flat_risk()
    optimize_zones()
    multi_attempt_analysis()
