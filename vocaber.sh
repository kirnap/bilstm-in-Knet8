#!/usr/bin/env bash

cat $1 | tr ' ' '\n' | sort | uniq -c | sort -nr
