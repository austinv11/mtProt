#!/usr/bin/env sh

rm -rf /workdir/aav4003/mtProtDocker
mkdir -p /workdir/aav4003/mtProtDocker
cp -r ./* /workdir/aav4003/mtProtDocker/

docker1 build --no-cache -t prot-docker /workdir/aav4003/mtProtDocker
#docker1 build -t prot-docker /workdir/aav4003/mtProtDocker
