# cipher/tweet_parser.py
#
# Handles multiple Twitter API response shapes that appear in the Android app:
#
#  1. GraphQL HomeTimeline / UserTweets  — deeply nested tweet_results tree
#  2. LivePipeline WebSocket events      — prefixed with "2<id>\n{...}"
#  3. v1-style push events               — tweet_create_events array
#  4. Thin tweet objects                 — direct {id_str, full_text, user{…}}
#
# Every successful parse returns a list of plain dicts:
#   { screen_name, text, id, created_at, ts }

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("cipher.parser")


# ── public entry-point ────────────────────────────────────────────────────────

def parse_tweets(raw: bytes | str) -> list[dict]:
    """
    Parse raw bytes/str from any Twitter API surface.
    Returns a (possibly empty) list of tweet dicts.
    """
    text = _to_str(raw)
    if not text:
        return []

    tweets: list[dict] = []

    # WebSocket LivePipeline envelope: "2<sub_id>\n{...}"
    for segment in _split_live_pipeline(text):
        try:
            payload = json.loads(segment)
            tweets.extend(_extract_all(payload))
        except json.JSONDecodeError:
            pass

    if not tweets:
        # Plain JSON blob (GraphQL response body or REST)
        start = text.find("{")
        if start < 0:
            start = text.find("[")
        if start >= 0:
            try:
                payload = json.loads(text[start:])
                tweets.extend(_extract_all(payload))
            except json.JSONDecodeError:
                pass

    # Stamp with wall-clock receipt time
    now = datetime.now(timezone.utc).isoformat()
    for t in tweets:
        t.setdefault("ts", now)

    return tweets


# ── LivePipeline frame splitting ──────────────────────────────────────────────

def _split_live_pipeline(text: str) -> list[str]:
    """
    LivePipeline frames look like:
        2<subscription_id>\n{"..."}
    or just bare JSON.  Return a list of candidate JSON strings.
    """
    segments: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip numeric/alpha subscription prefix before '{'
        brace = line.find("{")
        bracket = line.find("[")
        start = -1
        if brace >= 0 and (bracket < 0 or brace < bracket):
            start = brace
        elif bracket >= 0:
            start = bracket
        if start >= 0:
            segments.append(line[start:])
        else:
            segments.append(line)
    return segments if segments else [text]


# ── recursive extraction dispatch ─────────────────────────────────────────────

def _extract_all(obj: Any) -> list[dict]:
    tweets: list[dict] = []

    if isinstance(obj, list):
        for item in obj:
            tweets.extend(_extract_all(item))
        return tweets

    if not isinstance(obj, dict):
        return tweets

    # v1-style: tweet_create_events
    if "tweet_create_events" in obj:
        for raw_tweet in obj["tweet_create_events"]:
            t = _from_v1(raw_tweet)
            if t:
                tweets.append(t)
        return tweets

    # GraphQL wrapper: data.home_timeline_urt / data.user.result.timeline_v2 …
    if "data" in obj:
        tweets.extend(_extract_all(obj["data"]))

    # timeline_v2 / timeline entries
    for key in ("timeline_v2", "timeline", "home_timeline_urt"):
        if key in obj:
            tweets.extend(_walk_timeline(obj[key]))

    # User node
    if "user" in obj and isinstance(obj["user"], dict):
        tweets.extend(_extract_all(obj["user"]))

    # Result nodes (GraphQL union types)
    if "result" in obj and isinstance(obj["result"], dict):
        tweets.extend(_extract_all(obj["result"]))

    # tweet_results is the GraphQL wrapper around a single tweet
    if "tweet_results" in obj and isinstance(obj["tweet_results"], dict):
        t = _from_tweet_results(obj["tweet_results"])
        if t:
            tweets.append(t)

    # Direct legacy tweet object sitting at top level
    if "legacy" in obj and "full_text" in obj.get("legacy", {}):
        t = _from_result_node(obj)
        if t:
            tweets.append(t)

    # Payload envelope (LivePipeline)
    if "payload" in obj:
        tweets.extend(_extract_all(obj["payload"]))

    # items / entries arrays
    for key in ("items", "entries", "instructions"):
        if key in obj and isinstance(obj[key], list):
            for item in obj[key]:
                tweets.extend(_extract_all(item))

    return tweets


# ── timeline walker ───────────────────────────────────────────────────────────

def _walk_timeline(tl: Any) -> list[dict]:
    tweets: list[dict] = []
    if not isinstance(tl, dict):
        return tweets

    instructions = tl.get("instructions", [])
    for instr in instructions:
        if not isinstance(instr, dict):
            continue
        entries = instr.get("entries", [])
        for entry in entries:
            tweets.extend(_walk_entry(entry))
        # Some instructions have a single "entry"
        if "entry" in instr:
            tweets.extend(_walk_entry(instr["entry"]))
    return tweets


def _walk_entry(entry: Any) -> list[dict]:
    tweets: list[dict] = []
    if not isinstance(entry, dict):
        return tweets

    content = entry.get("content", {})
    if not isinstance(content, dict):
        return tweets

    # itemContent or items list
    item_content = content.get("itemContent", {})
    if item_content:
        tweets.extend(_from_item_content(item_content))

    items = content.get("items", [])
    for item in items:
        tweets.extend(_from_item_content(item.get("item", {}).get("itemContent", {})))

    return tweets


def _from_item_content(ic: Any) -> list[dict]:
    if not isinstance(ic, dict):
        return []
    if ic.get("itemType") not in (
        "TimelineTweet",
        "TimelineTimelineTweet",
        None,  # fallback
    ):
        # Still try if tweet_results present
        if "tweet_results" not in ic:
            return []
    tr = ic.get("tweet_results", {})
    t = _from_tweet_results(tr)
    return [t] if t else []


# ── tweet extraction helpers ──────────────────────────────────────────────────

def _from_tweet_results(tr: dict) -> dict | None:
    result = tr.get("result", {})
    if not result:
        return None
    # Handle __typename: TweetWithVisibilityResults wrapper
    if result.get("__typename") == "TweetWithVisibilityResults":
        result = result.get("tweet", result)
    return _from_result_node(result)


def _from_result_node(result: dict) -> dict | None:
    legacy = result.get("legacy", {})
    if not legacy or "full_text" not in legacy:
        return None

    core = result.get("core", {})
    user_result = core.get("user_results", {}).get("result", {})
    user_legacy = user_result.get("legacy", {})

    screen_name = user_legacy.get("screen_name") or legacy.get("user", {}).get(
        "screen_name", ""
    )

    return _build(
        screen_name=screen_name,
        text=legacy.get("full_text", ""),
        tweet_id=legacy.get("id_str", result.get("rest_id", "")),
        created_at=legacy.get("created_at", ""),
    )


def _from_v1(raw: dict) -> dict | None:
    """Parse a v1.1 tweet object (tweet_create_events)."""
    if not isinstance(raw, dict):
        return None
    user = raw.get("user", {})
    return _build(
        screen_name=user.get("screen_name", ""),
        text=raw.get("full_text") or raw.get("text", ""),
        tweet_id=raw.get("id_str", str(raw.get("id", ""))),
        created_at=raw.get("created_at", ""),
    )


def _build(
    screen_name: str, text: str, tweet_id: str, created_at: str
) -> dict | None:
    if not screen_name or not text:
        return None
    return {
        "screen_name": screen_name,
        "text": text,
        "id": tweet_id,
        "created_at": created_at,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_str(raw: bytes | str) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="ignore")
    return raw or ""
