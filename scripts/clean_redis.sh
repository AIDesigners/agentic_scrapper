#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# clean_redis.sh - purge the crawler's visited-URL set for a fresh start.
#
# Deletes the Redis key that tracks already-visited URLs so a new crawler
# run starts with a clean slate. The key itself is removed (the next run
# recreates it on the first check_and_add). Targeted DEL, NOT FLUSHDB:
# any other keys in the database are left untouched.
#
# Defaults are overridable via environment variables:
#   REDIS_HOST  (default: localhost)
#   REDIS_PORT  (default: 6379)
#   REDIS_DB    (default: 0)
#   REDIS_KEY   (default: crawler_visited_urls - see RedisDBClient)
#
# Usage:
#   ./scripts/clean_redis.sh
#   REDIS_KEY=other_key ./scripts/clean_redis.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"
REDIS_KEY="${REDIS_KEY:-crawler_visited_urls}"

rcli() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" "$@"
}

before="$(rcli SCARD "$REDIS_KEY")"
rcli DEL "$REDIS_KEY" >/dev/null
after="$(rcli SCARD "$REDIS_KEY")"

echo "Redis ${REDIS_HOST}:${REDIS_PORT} db=${REDIS_DB} key=${REDIS_KEY}: members ${before} -> ${after}"
