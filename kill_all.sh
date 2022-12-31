#!/usr/bin/env sh

docker1 ps | grep "$(whoami)" | awk '{print $1}' | xargs docker1 kill