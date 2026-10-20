#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# clean_rabbitmq.sh - purge messages from the crawler queue for a fresh start.
#
# Drops all messages sitting in the crawler's publish queue so a new run
# starts with an empty queue. The queue definition itself is KEPT (the
# crawler never declares it - it must already exist on the broker).
# Uses the RabbitMQ management HTTP API via rabbitmqadmin.
#
# NOTE: the management API mirrors queue state with a delay of a few
# seconds, so the post-purge count is polled (up to 15 s) until it settles
# to 0 instead of being trusted on a single read.
#
# Defaults are overridable via environment variables:
#   RABBITMQ_HOST   (default: localhost)
#   RABBITMQ_PORT   (default: 15672 - management API port, NOT 5672 AMQP)
#   RABBITMQ_VHOST  (default: /)
#   RABBITMQ_USER   (default: guest)
#   RABBITMQ_PASS   (default: guest)
#   RABBITMQ_QUEUE  (default: crawler_json_queue - see crawler wiring)
#
# Usage:
#   ./scripts/clean_rabbitmq.sh
#   RABBITMQ_QUEUE=scrapper_json_queue ./scripts/clean_rabbitmq.sh
# ---------------------------------------------------------------------------
set -euo pipefail

RABBITMQ_HOST="${RABBITMQ_HOST:-localhost}"
RABBITMQ_PORT="${RABBITMQ_PORT:-15672}"
RABBITMQ_VHOST="${RABBITMQ_VHOST:-/}"
RABBITMQ_USER="${RABBITMQ_USER:-guest}"
RABBITMQ_PASS="${RABBITMQ_PASS:-guest}"
RABBITMQ_QUEUE="${RABBITMQ_QUEUE:-crawler_json_queue}"

# rabbitmqadmin flags: -H host, -P management port, -V vhost, -u user, -p password
rma() {
    rabbitmqadmin -q -f tsv -H "$RABBITMQ_HOST" -P "$RABBITMQ_PORT" -V "$RABBITMQ_VHOST" \
                  -u "$RABBITMQ_USER" -p "$RABBITMQ_PASS" "$@"
}

queue_messages() {
    rma list queues name messages | awk -v q="$RABBITMQ_QUEUE" '$1 == q {print $2}'
}

before="$(queue_messages)"
if [[ -z "$before" ]]; then
    echo "ERROR: queue '${RABBITMQ_QUEUE}' not found in vhost '${RABBITMQ_VHOST}' on ${RABBITMQ_HOST}:${RABBITMQ_PORT}" >&2
    exit 1
fi

rma purge queue name="${RABBITMQ_QUEUE}"

# Poll until the (lagging) management API settles on 0 messages.
after=""
for _ in $(seq 1 15); do
    after="$(queue_messages)"
    if [[ "$after" == "0" ]]; then
        break
    fi
    sleep 1
done

if [[ "$after" != "0" ]]; then
    echo "ERROR: queue '${RABBITMQ_QUEUE}' still reports ${after:-unknown} messages after purge" >&2
    exit 1
fi

echo "RabbitMQ ${RABBITMQ_HOST}:${RABBITMQ_PORT} vhost=${RABBITMQ_VHOST} queue=${RABBITMQ_QUEUE}: messages ${before} -> ${after}"
