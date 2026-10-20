#!/bin/bash

# Show all queues
rabbitmqadmin list queues name messages_ready messages_unacknowledged messages consumers

## Inspect particular queue
#rabbitmqadmin get queue="scrapper_json_queue" ackmode=ack_requeue_true count=5