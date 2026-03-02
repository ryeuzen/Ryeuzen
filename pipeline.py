#!/usr/bin/env python3
# cipher/pipeline.py  —  central async pipeline
#
# Architecture:
#   interceptor.py (mitmproxy) ──TCP 9999──► pipeline.py ──► classifier
#                                                          └──► market_matcher (stub)
#
# Run:  python3 pipeline.py

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from classifier import Signal, classify
from config import LOG_DATE_FORMAT, LOG_FORMAT, PIPELINE_HOST, PIPELINE_PORT

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
log = logging.getLogger("cipher.pipeline")

# ── in-memory signal log (last 1 000 signals, for debugging) ──────────────────
_signal_log: deque[dict] = deque(maxlen=1000)
# de-dup: ignore repeat tweets within 30 s
_seen_ids: dict[str, float] = {}
_DEDUP_TTL = 30.0  # seconds


# ── connection handler ─────────────────────────────────────────────────────────

async def _handle_interceptor(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    addr = writer.get_extra_info("peername")
    log.info("Interceptor connected from %s", addr)
    t0 = time.monotonic()
    tweets_in = 0
    signals_out = 0

    try:
        async for raw_line in reader:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                tweet = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                log.warning("Bad JSON: %s — %s", exc, raw_line[:120])
                continue

            tweets_in += 1
            tweet_id = tweet.get("id", "")

            # de-duplicate
            now = time.monotonic()
            if tweet_id:
                if tweet_id in _seen_ids and (now - _seen_ids[tweet_id]) < _DEDUP_TTL:
                    continue
                _seen_ids[tweet_id] = now
                # prune old entries
                if len(_seen_ids) > 5000:
                    cutoff = now - _DEDUP_TTL
                    expired = [k for k, v in _seen_ids.items() if v < cutoff]
                    for k in expired:
                        del _seen_ids[k]

            sig = classify(tweet)
            if sig:
                signals_out += 1
                _record_signal(sig)
                await _dispatch(sig)

    except asyncio.IncompleteReadError:
        pass
    except Exception as exc:
        log.exception("Unexpected error in handler: %s", exc)
    finally:
        elapsed = time.monotonic() - t0
        log.info(
            "Interceptor disconnected %s  tweets=%d  signals=%d  uptime=%.0fs",
            addr,
            tweets_in,
            signals_out,
            elapsed,
        )
        writer.close()


# ── signal recording ───────────────────────────────────────────────────────────

def _record_signal(sig: Signal) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "ts": ts,
        "account": sig.account,
        "category": sig.category,
        "confidence": sig.confidence,
        "keywords": sig.keywords_matched,
        "tweet_id": sig.tweet_id,
        "text": sig.tweet_text,
    }
    _signal_log.append(entry)
    log.info(
        "SIGNAL  [%s]  @%-18s  conf=%.2f  kw=%s",
        sig.category,
        sig.account,
        sig.confidence,
        sig.keywords_matched,
    )
    log.info("        %s", sig.tweet_text[:200])


# ── downstream dispatch (stub for market matcher) ──────────────────────────────

async def _dispatch(sig: Signal) -> None:
    """
    Phase 2 stub:  rapidfuzz match against live Polymarket CLOB markets,
    then route to order executor.

    Replace this function body with:
        from market_matcher import match_and_execute
        await match_and_execute(sig)
    """
    log.info("        → market_matcher (stub)  — signal queued for Phase 2")


# ── graceful shutdown ─────────────────────────────────────────────────────────

def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def _stop():
        log.info("Shutdown requested — stopping pipeline")
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows
            signal.signal(sig, lambda *_: _stop())


# ── main ───────────────────────────────────────────────────────────────────────

async def _main() -> None:
    server = await asyncio.start_server(
        _handle_interceptor, PIPELINE_HOST, PIPELINE_PORT
    )
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    log.info("Pipeline listening on %s", addrs)
    log.info("Waiting for mitmproxy interceptor to connect…")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)
    try:
        loop.run_until_complete(_main())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.close()
        log.info("Pipeline stopped. Signals captured: %d", len(_signal_log))
