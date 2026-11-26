#!/bin/bash

PG_DATA="/home/ayakovenko/psql/"

initdb -D "${PG_DATA}" -U postgres --auth=trust

echo "shared_preload_libraries = 'pg_textsearch'" >> "${PG_DATA}/postgresql.conf"

pg_ctl -D "${PG_DATA}" -l "${PG_DATA}/postgres.log" start

createdb -U postgres rag_db
psql -U postgres -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_textsearch;"
psql -U postgres -d rag_db -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_textsearch');"

pg_ctl -D "${PG_DATA}" stop

