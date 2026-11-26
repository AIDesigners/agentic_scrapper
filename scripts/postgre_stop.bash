#!/bin/bash

PG_DATA="/home/ayakovenko/psql/"


#pg_ctl -D "${PG_DATA}" -l "${PG_DATA}/postgres.log" start

pg_ctl -D "${PG_DATA}" stop

