#!/usr/bin/env sh

docker1 ps | grep "$(whoami)" | grep "prot_docker" | awk '{print $1}' | xargs docker1 kill