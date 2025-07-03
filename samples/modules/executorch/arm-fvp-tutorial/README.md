# Arm FVP Tutorial

This tutorial is to demonstrate how to run an exported Executorch model in the Arm FVP simulator.

# Setup

Its strongly recommended to use the docker image paired with this tutorial. You can do this by either buildin gthe image directly, or pull frim Dockerhub.

## Building the Image from source

## Pulling the container

## Starting the docker image

# Wokring inside the Docker image

## Setup Repo

```
git clone https://github.com/BujSet/zephyr.git
cd zephyr/
git switch -c executorch-module-integration origin/executorch-module-integration
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install west torch numpy
python3 -m pip install -r scripts/requirements.txt
cd ../
west init -l zephyr
west config manifest.project-filter -- +executorch
west -v update
source zephyr/zephyr-env.sh
export PYTHONPATH=/home/zephyruser/modules/lib
```

## Building the elf

```
cd /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world
west build -p always -b mps3/corstone300/an547

```
