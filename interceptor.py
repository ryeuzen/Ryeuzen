# cipher/interceptor.py  —  mitmproxy addon
#
# Run with:
#   mitmdump -s interceptor.py --mode regular -p 8080 --ssl-insecure
#
# Intercepts ALL traffic from the Twitter/X Android app:
#   • HTTP(S) GraphQL responses  (HomeTimeline, UserTweets, …)
#   • WebSocket LivePipeline frames (wss://api.twitter.com/2/live_pipeline/events)
#
# Tweets from TARGET_ACCOUNTS are forwarded as newline-delimited JSON to the
# pipeline server (pipeline.py) listening on 127.0.0.1:9999.

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Optional

from mitmproxy import ctx, http
from mitmproxy.websocket import WebSocketMessage

from config import (
    GRAPHQL_TWEET_OPS,
    MITM_HOST,
    PIPELINE_HOST,
    PIPELINE_PORT,
    TARGET_ACCOUNTS_LOWER,
    TWITTER_HOSTS,
    WS_LIVE_PATHS,
)
from tweet_parser import parse_tweets

log = logging.getLogger("cipher.interceptor")


class _PipelineSocket:
    """
    Persistent TCP connection to pipeline.py.
    Thread-safe; auto-reconnects on failure.
    """

    RETRY_DELAY = 0.5  # seconds between reconnect attempts

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((self._host, self._port))
            s.settimeout(None)
            self._sock = s
            log.info("Connected to pipeline %s:%d", self._host, self._port)
            return True
        except OSError as exc:
            log.warning("Pipeline connect failed: %s", exc)
            self._sock = None
            return False

    def send(self, tweet: dict) -> None:
        line = (json.dumps(tweet, ensure_ascii=False) + "\n").encode()
        with self._lock:
            for attempt in range(3):
                if self._sock is None and not self._connect():
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                try:
                    self._sock.sendall(line)  # type: ignore[union-attr]
                    return
                except OSError:
                    self._sock = None
            log.error("Dropped tweet (pipeline unreachable): %s", tweet.get("id"))


# Singleton socket shared across all mitmproxy worker threads
_pipe: Optional[_PipelineSocket] = None
_pipe_lock = threading.Lock()


def _get_pipe() -> _PipelineSocket:
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                _pipe = _PipelineSocket(PIPELINE_HOST, PIPELINE_PORT)
    return _pipe


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_twitter(host: str) -> bool:
    return any(th in host for th in TWITTER_HOSTS)


def _is_graphql_tweet_op(path: str) -> bool:
    # path looks like: /graphql/<hash>/HomeTimeline
    parts = path.rstrip("/").rsplit("/", 1)
    return parts[-1] in GRAPHQL_TWEET_OPS if parts else False


def _is_live_pipeline(path: str) -> bool:
    return any(path.startswith(p) for p in WS_LIVE_PATHS)


def _dispatch(tweets: list[dict]) -> None:
    pipe = _get_pipe()
    for tweet in tweets:
        if tweet.get("screen_name", "").lower() in TARGET_ACCOUNTS_LOWER:
            log.debug(
                "→ @%s: %.80s", tweet["screen_name"], tweet.get("text", "")
            )
            pipe.send(tweet)


# ── mitmproxy addon ───────────────────────────────────────────────────────────

class CipherInterceptor:
    # ── HTTP responses ─────────────────────────────────────────────────────────

    def response(self, flow: http.HTTPFlow) -> None:
        host = flow.request.pretty_host
        if not _is_twitter(host):
            return

        path = flow.request.path
        if not (_is_graphql_tweet_op(path) or _is_live_pipeline(path)):
            return

        if flow.response is None:
            return

        ct = flow.response.headers.get("content-type", "")
        if "json" not in ct and "stream" not in ct:
            return

        try:
            body = flow.response.get_content()
            if not body:
                return
            tweets = parse_tweets(body)
            _dispatch(tweets)
        except Exception as exc:
            log.exception("Error in response handler: %s", exc)

    # ── WebSocket frames ───────────────────────────────────────────────────────

    def websocket_message(self, flow: http.HTTPFlow) -> None:
        host = flow.request.pretty_host
        if not _is_twitter(host):
            return

        msg: WebSocketMessage = flow.websocket.messages[-1]  # type: ignore[union-attr]
        if msg.from_client:
            return  # only server-pushed frames carry tweet data

        try:
            tweets = parse_tweets(msg.content)
            _dispatch(tweets)
        except Exception as exc:
            log.exception("Error in websocket handler: %s", exc)


addons = [CipherInterceptor()]
