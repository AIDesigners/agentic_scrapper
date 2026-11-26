#!/bin/bash

PREFIX="/home/ayakovenko/.local/"


cd ./postgresql-18.1/
./configure --prefix=${PREFIX}
make -j 8
make install
cd ../

cd ./pgvector/
make PG_CONFIG="${PREFIX}/bin/pg_config"
make PG_CONFIG="${PREFIX}/bin/pg_config" install
cd ../
cd ./pg_textsearch/
make PG_CONFIG="${PREFIX}/bin/pg_config"
make PG_CONFIG="${PREFIX}/bin/pg_config" install
cd ../



