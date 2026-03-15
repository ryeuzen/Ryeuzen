"""
Configuration for prop firm challenge simulation.

The core insight: zero-EV strategies become +EV when wrapped in the
convex payoff structure of prop firm challenges.

Challenge structure:
  - Pay fixed fee F to attempt
  - Hit profit target P% before hitting max drawdown D%
  - If funded: collect payout_split * profits, lose account at drawdown

This is equivalent to buying a call option where:
  - Premium = challenge fee
  - Strike = profit target
  - Underlying = your P&L path

Zone-based sizing adapts risk fraction based on where equity sits
relative to the target and drawdown floor:

  ┌─────────────────────────┐  ← profit target (108k)
  │  CRUISE zone            │  risk_cruise (conservative, protect gains)
  ├─────────────────────────┤  ← cruise_threshold (e.g. 75% of target)
  │  NORMAL zone            │  risk_normal (standard 2%)
  ├─────────────────────────┤  ← initial balance (100k)
  │  CAUTION zone           │  risk_caution (reduced, extend runway)
  ├─────────────────────────┤  ← hail_mary_threshold (e.g. 3% from DD)
  │  HAIL MARY zone         │  risk_hail_mary (max aggression, go big)
  ├─────────────────────────┤  ← daily stop threshold
  └─────────────────────────┘  ← max drawdown floor (90k)

The daily drawdown reset is the constraint that kills most attempts.
Each day starts fresh — a single bad day can end the challenge even if
your overall drawdown is fine.
"""

# ── Challenge parameters ────────────────────────────────────────────
CHALLENGE_FEE = 500            # USD cost per challenge attempt
ACCOUNT_SIZE = 100_000         # Simulated account size
PROFIT_TARGET_PCT = 0.08       # 8% profit target to pass
MAX_DRAWDOWN_PCT = 0.10        # 10% max drawdown (from initial balance)
MAX_DAILY_DRAWDOWN_PCT = 0.05  # 5% max daily drawdown
MIN_TRADING_DAYS = 5           # Minimum days before passing
MAX_TRADING_DAYS = 30          # Calendar days to complete challenge
TRADES_PER_DAY = 3             # Average trades per session

# ── Strategy parameters (zero-EV ORB) ─────────────────────────────
# Default win rate and reward:risk that yield EV = 0
# EV = WR * RR - (1 - WR) * 1.0 = 0
# With WR = 2/3, RR = 0.5: EV = 0.6667*0.5 - 0.3333*1.0 = 0
DEFAULT_WIN_RATE = 2 / 3       # 66.67%
DEFAULT_REWARD_RISK = 0.5      # Win $0.50 per $1 risked

# ── Zone-based sizing (challenge phase) ────────────────────────────
# Zone thresholds (as fraction of distance from floor to target)
#
# cruise_threshold: fraction of profit target reached before switching
#   to conservative sizing. E.g. 0.75 = switch at 75% of target.
# hail_mary_threshold: remaining buffer (as fraction of max DD) below
#   which we go maximum aggression. E.g. 0.30 = if only 30% of DD
#   buffer remains, go all-in.
# daily_stop_pct: fraction of daily DD limit used before stopping for
#   the day. E.g. 0.70 = stop after using 70% of daily DD.

ZONE_SIZING = {
    "risk_normal":   0.030,    # 3.0% — standard zone (equity ~= initial)
    "risk_cruise":   0.015,    # 1.5% — near target, protect gains
    "risk_caution":  0.020,    # 2.0% — in drawdown, don't slow recovery
    "risk_hail_mary": 0.060,   # 6.0% — near death, nothing to lose

    # Zone boundaries
    "cruise_threshold": 0.75,  # Switch to cruise at 75% of target
    "hail_mary_threshold": 0.35,  # Hail Mary when ≤35% DD buffer left
    "daily_stop_pct": 0.50,    # Stop trading after 50% daily DD used
}

# ── Phase 1: frac_2pct (flat baseline for comparison) ─────────────
# Fixed fractional 2% risk per trade, no zone logic
FRAC_2PCT = {
    "name": "frac_2pct",
    "description": "Fixed fractional 2% risk, ORB zero-EV",
    "risk_per_trade": 0.02,
    "win_rate": DEFAULT_WIN_RATE,
    "reward_risk_ratio": DEFAULT_REWARD_RISK,
    "sizing": "fixed_fractional",
}

# ── Phase 2: dd_frac_40pct (funded phase) ─────────────────────────
# Drawdown-fractional 40%: risk 40% of remaining drawdown buffer
# Same zero-EV ORB parameters, but aggressive sizing
DD_FRAC_40PCT = {
    "name": "dd_frac_40pct",
    "description": "Drawdown-fractional 40%, maximize funded payout",
    "dd_fraction": 0.40,
    "win_rate": DEFAULT_WIN_RATE,
    "reward_risk_ratio": DEFAULT_REWARD_RISK,
    "sizing": "dd_fractional",
}

# ── Funded account parameters ─────────────────────────────────────
FUNDED_MAX_DRAWDOWN_PCT = 0.10   # 10% max drawdown on funded account
FUNDED_PAYOUT_SPLIT = 0.80       # Trader keeps 80% of profits
FUNDED_PAYOUT_THRESHOLD = 1000   # Min profit to request payout
FUNDED_MAX_TRADES = 200          # Max trades before forcing payout eval

# ── Monte Carlo simulation ────────────────────────────────────────
MC_SIMULATIONS = 100_000         # Number of Monte Carlo paths
MC_SEED = 42                     # Reproducibility
