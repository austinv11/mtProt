#!/usr/bin/env sh

#docker1 stop aav4003__biohpc_prot-docker
#docker1 rm aav4003__biohpc_prot-docker

# docker1 run --gpus all -d biohpc_aav4003/prot-docker:latest
docker1 run --gpus "\"device=1\"" -d "biohpc_aav4003/prot_docker_vanilla_sweep:latest"
docker1 run --gpus "\"device=2\"" -d "biohpc_aav4003/prot_docker_sparse_sweep:latest"
docker1 run --gpus "\"device=1\"" -d "biohpc_aav4003/prot_docker_contractive_sweep:latest"
docker1 run --gpus "\"device=2\"" -d "biohpc_aav4003/prot_docker_concrete_sweep:latest"