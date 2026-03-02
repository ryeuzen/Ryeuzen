# cipher/config.py — central configuration

# ── Target accounts ────────────────────────────────────────────────────────────
TARGET_ACCOUNTS = [
    "ShamsCharania",
    "wojespn",
    "MarcJSpears",
    "ChrisBHaynes",
    "FabrizioRomano",
    "David_Ornstein",
    "SkySportsNews",
]

# Lowercase set for fast membership test
TARGET_ACCOUNTS_LOWER = {a.lower() for a in TARGET_ACCOUNTS}

# ── Android / ADB ─────────────────────────────────────────────────────────────
AVD_NAME = "cipher_pixel6"
EMULATOR_SERIAL = "emulator-5554"

# ── mitmproxy ─────────────────────────────────────────────────────────────────
MITM_HOST = "127.0.0.1"
MITM_PORT = 8080

# ── Inter-process pipeline (interceptor → pipeline server) ────────────────────
PIPELINE_HOST = "127.0.0.1"
PIPELINE_PORT = 9999

# ── Twitter hosts to intercept ────────────────────────────────────────────────
TWITTER_HOSTS = {
    "api.twitter.com",
    "api.x.com",
    "twitter.com",
    "x.com",
}

# WebSocket paths that carry real-time events
WS_LIVE_PATHS = {
    "/2/live_pipeline/events",
    "/live_pipeline/events",
}

# GraphQL operation names that return timeline tweets
GRAPHQL_TWEET_OPS = {
    "HomeTimeline",
    "HomeLatestTimeline",
    "UserTweets",
    "UserTweetsAndReplies",
    "TweetDetail",
    "SearchTimeline",
    "Following",
    "FollowersYouKnow",
    "BookmarksTimeline",
    "ListLatestTweetsTimeline",
}

# ── Classifier thresholds ─────────────────────────────────────────────────────
MIN_SIGNAL_CONFIDENCE = 0.25  # drop signals below this confidence

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s.%(msecs)03d  %(levelname)-7s %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
