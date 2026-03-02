# cipher/classifier.py  —  signal classifier
#
# Accepts a parsed tweet dict and returns a Signal (or None).
# Covers NBA injury reports and EPL lineup/transfer news.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from config import MIN_SIGNAL_CONFIDENCE, TARGET_ACCOUNTS_LOWER

# ── keyword tables ─────────────────────────────────────────────────────────────

_NBA_INJURY_KW = [
    "out", "ruled out", "will not play", "won't play", "questionable",
    "doubtful", "probable", "day-to-day", "game-time decision", "gtd",
    "did not play", "dnp", "miss", "missed", "missing",
    "sidelined", "sidelining", "sidelined indefinitely",
    "knee", "ankle", "hamstring", "quad", "calf", "achilles",
    "back", "shoulder", "wrist", "hand", "finger", "elbow",
    "hip", "groin", "foot", "concussion", "illness",
    "injury", "injured", "strain", "sprain", "fracture",
    "return", "returns", "returned", "cleared", "available",
    "trade", "traded", "waived", "released", "suspended",
    "suspension", "ejected",
]

_EPL_LINEUP_KW = [
    "lineup", "xi", "starting xi", "starting lineup",
    "bench", "squad", "team news", "dropped",
    "suspended", "suspension", "banned", "unavailable",
    "ruled out", "out", "misses", "miss", "missing",
    "injured", "injury", "fit", "fitness", "fitness test",
    "doubtful", "doubt", "available", "returns",
    "transfer", "signed", "signs", "deal", "loan",
    "confirmed", "announced", "official",
]

_NBA_TEAMS = [
    "lakers", "celtics", "warriors", "nets", "bucks", "heat", "suns",
    "nuggets", "clippers", "sixers", "76ers", "knicks", "mavs",
    "mavericks", "grizzlies", "pelicans", "hawks", "bulls", "raptors",
    "spurs", "rockets", "jazz", "thunder", "blazers", "trail blazers",
    "kings", "magic", "pistons", "pacers", "hornets", "wizards",
    "cavaliers", "cavs", "timberwolves", "wolves", "nba",
]

_EPL_TEAMS = [
    "arsenal", "chelsea", "liverpool", "manchester city", "man city",
    "manchester united", "man utd", "man united", "tottenham", "spurs",
    "newcastle", "aston villa", "west ham", "everton", "brighton",
    "brentford", "fulham", "crystal palace", "nottingham forest",
    "bournemouth", "wolves", "wolverhampton", "leicester", "ipswich",
    "southampton", "premier league", "epl",
]

# Pre-lower-cased for fast matching
_NBA_INJURY_KW_L = [k.lower() for k in _NBA_INJURY_KW]
_EPL_LINEUP_KW_L = [k.lower() for k in _EPL_LINEUP_KW]
_NBA_TEAMS_L = [t.lower() for t in _NBA_TEAMS]
_EPL_TEAMS_L = [t.lower() for t in _EPL_TEAMS]


# ── Signal dataclass ───────────────────────────────────────────────────────────

@dataclass
class Signal:
    account: str
    tweet_id: str
    tweet_text: str
    category: str                      # "nba_injury" | "epl_lineup"
    keywords_matched: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_tweet: dict = field(default_factory=dict)


# ── public API ─────────────────────────────────────────────────────────────────

def classify(tweet: dict) -> Optional[Signal]:
    """
    Return a Signal if the tweet from a target account contains a tradeable
    signal, otherwise None.
    """
    screen_name = tweet.get("screen_name", "")
    if screen_name.lower() not in TARGET_ACCOUNTS_LOWER:
        return None

    text = tweet.get("text", "")
    text_l = text.lower()
    tweet_id = tweet.get("id", "")

    nba_sig = _score_nba(text_l)
    epl_sig = _score_epl(text_l)

    # Pick the higher-confidence signal
    best_cat, best_kw, best_conf = None, [], 0.0
    if nba_sig[1] > best_conf:
        best_cat, best_kw, best_conf = "nba_injury", nba_sig[0], nba_sig[1]
    if epl_sig[1] > best_conf:
        best_cat, best_kw, best_conf = "epl_lineup", epl_sig[0], epl_sig[1]

    if best_cat is None or best_conf < MIN_SIGNAL_CONFIDENCE:
        return None

    return Signal(
        account=screen_name,
        tweet_id=tweet_id,
        tweet_text=text,
        category=best_cat,
        keywords_matched=best_kw,
        confidence=round(best_conf, 3),
        raw_tweet=tweet,
    )


# ── scoring ────────────────────────────────────────────────────────────────────

def _score_nba(text_l: str) -> tuple[list[str], float]:
    kw_hit = [kw for kw in _NBA_INJURY_KW_L if kw in text_l]
    team_hit = any(t in text_l for t in _NBA_TEAMS_L)
    if not kw_hit or not team_hit:
        return [], 0.0
    # Base: 0.25 for team + 0.15 per keyword, max 1.0
    conf = min(1.0, 0.25 + len(kw_hit) * 0.15)
    # Boost for high-urgency terms
    urgent = {"ruled out", "out", "will not play", "won't play", "dnp"}
    if urgent & set(kw_hit):
        conf = min(1.0, conf + 0.20)
    return kw_hit, conf


def _score_epl(text_l: str) -> tuple[list[str], float]:
    kw_hit = [kw for kw in _EPL_LINEUP_KW_L if kw in text_l]
    team_hit = any(t in text_l for t in _EPL_TEAMS_L)
    if not kw_hit or not team_hit:
        return [], 0.0
    conf = min(1.0, 0.25 + len(kw_hit) * 0.15)
    urgent = {"ruled out", "out", "signed", "official", "confirmed"}
    if urgent & set(kw_hit):
        conf = min(1.0, conf + 0.20)
    return kw_hit, conf
