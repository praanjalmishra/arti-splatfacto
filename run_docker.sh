#!/bin/bash

# Create necessary directories
mkdir -p ~/arti-outputs ~/nerfstudio-data

# Run container interactively
docker compose run --rm --service-ports arti-splatfacto bash