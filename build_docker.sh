#!/usr/bin/env sh

rm -rf /workdir/aav4003/mtProtDocker
mkdir -p /workdir/aav4003/mtProtDocker
cp -r ./* /workdir/aav4003/mtProtDocker/

# NOTE: Pass the sweep file as an argument to this script

docker1 build --no-cache -t "prot_docker_$1" --build-arg sweep_file="$1.yaml" /workdir/aav4003/mtProtDocker
#docker1 build -t prot-docker /workdir/aav4003/mtProtDocker
