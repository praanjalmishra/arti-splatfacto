#!/bin/bash
docker run --gpus all \
    -e USER=pmishra \
    -v /local/home/pmishra:/workspace/ \
    -v /local/home/pmishra/cache/:/home/user/.cache/ \
    -p 7007:7007 \
    --rm \
    -it \
    arti-splatfacto:latest