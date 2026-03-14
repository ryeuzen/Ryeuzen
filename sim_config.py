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

# ── Phase 1: frac_2pct (challenge phase) ────────────────────────────
# Fixed fractional 2% risk per trade
# ORB (Opening Range Breakout) parameters yielding zero EV:
#   Win rate = 2/3 ≈ 66.67%
#   Risk:Reward = 1:0.5 (risk $1 to make $0.50)
#   EV = 0.6667 * 0.5 - 0.3333 * 1.0 = 0.0
FRAC_2PCT = {
    "name": "frac_2pct",
    "description": "Fixed fractional 2% risk, ORB zero-EV",
    "risk_per_trade": 0.02,       # 2% of current equity risked
    "win_rate": 2 / 3,            # 66.67%
    "reward_risk_ratio": 0.5,     # Win $0.50 per $1 risked
    "sizing": "fixed_fractional", # f(equity) = equity * risk_per_trade
}

# ── Phase 2: dd_frac_40pct (funded phase) ───────────────────────────
# Drawdown-fractional 40%: risk 40% of remaining drawdown buffer
# Same zero-EV ORB parameters, but aggressive sizing
# Buffer = max_dd_limit - current_drawdown
# Risk per trade = 0.40 * buffer
DD_FRAC_40PCT = {
    "name": "dd_frac_40pct",
    "description": "Drawdown-fractional 40%, maximize funded payout",
    "dd_fraction": 0.40,          # 40% of remaining drawdown buffer
    "win_rate": 2 / 3,
    "reward_risk_ratio": 0.5,
    "sizing": "dd_fractional",    # f(buffer) = buffer * dd_fraction
}

# ── Funded account parameters ───────────────────────────────────────
FUNDED_MAX_DRAWDOWN_PCT = 0.10   # 10% max drawdown on funded account
FUNDED_PAYOUT_SPLIT = 0.80       # Trader keeps 80% of profits
FUNDED_PAYOUT_THRESHOLD = 1000   # Min profit to request payout
FUNDED_MAX_TRADES = 200          # Max trades before forcing payout eval

# ── Monte Carlo simulation ──────────────────────────────────────────
MC_SIMULATIONS = 100_000         # Number of Monte Carlo paths
MC_SEED = 42                     # Reproducibility
