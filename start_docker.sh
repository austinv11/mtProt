#!/usr/bin/env sh

docker1 stop aav4003__biohpc_prot-docker
docker1 rm aav4003__biohpc_prot-docker

docker1 run --gpus all --name prot-docker -d biohpc_aav4003/prot-docker:latest