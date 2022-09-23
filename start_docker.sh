#!/usr/bin/env sh

#docker1 stop aav4003__biohpc_prot-docker
#docker1 rm aav4003__biohpc_prot-docker

# docker1 run --gpus all -d biohpc_aav4003/prot-docker:latest
docker1 run --gpus all -d "biohpc_aav4003/prot_docker_$1:latest"