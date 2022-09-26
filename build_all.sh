#!/usr/bin/env sh

rm -rf /workdir/aav4003/mtProtDocker
mkdir -p /workdir/aav4003/mtProtDocker
cp -r ./* /workdir/aav4003/mtProtDocker/

docker1 build --no-cache -t "prot_docker_vanilla_sweep" --build-arg sweep_file="vanilla_sweep.yaml" /workdir/aav4003/mtProtDocker
docker1 build --no-cache -t "prot_docker_sparse_sweep" --build-arg sweep_file="sparse_sweep.yaml" /workdir/aav4003/mtProtDocker
docker1 build --no-cache -t "prot_docker_contractive_sweep" --build-arg sweep_file="contractive_sweep.yaml" /workdir/aav4003/mtProtDocker
docker1 build --no-cache -t "prot_docker_concrete_sweep" --build-arg sweep_file="concrete_sweep.yaml" /workdir/aav4003/mtProtDocker
