# Arm FVP Tutorial

This tutorial is to demonstrate how to run an exported Executorch model in the Arm FVP simulator.

# Setup

Its strongly recommended to use the docker image paired with this tutorial. You can do this by either buildin gthe image directly, or pull from Dockerhub.

## Building the Image from source

```
docker build -t rselagam/zephyr-armfvp:v2 -f Dockerfile.armfvp_zephyr  .
```

## Pulling the container

```
docker pull rselagam/zephyr-armfvp:v1
```

## Starting the docker image

### Linux/macOS

```
docker run --rm -it --entrypoint /bin/bash  --net=host -v "$(pwd)"/workspace:/workspace -w home/zephyruser/ rselagam/zephyr-armfvp:v1
```

### Windows (PowerShell)

```
docker run --rm -it --entrypoint /bin/bash --net=host -v "${PWD}\workspace:/workspace" -w /home/zephyruser/ rselagam/zephyr-armfvp:v1
```

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

## Setup FVP Simulator

```
cd /home/zephyruser/modules/lib/executorch/
./examples/arm/setup.sh --i-agree-to-the-contained-eula  --target-toolchain zephyr
source /home/zephyruser/modules/lib/executorch/examples/arm/ethos-u-scratch/setup_path.sh
```

## Building the elf

```
cd /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world
west build -p always -b mps3/corstone300/an547
```
## Running a model in the simulator

```
FVP_Corstone_SSE-300_Ethos-U55 -a /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/build/zephyr/zephyr.elf -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file='-' --simlimit 30
```

And you should see output like:


## [WIP] Quick start commands to repro the issue

Start the docker image:

```
docker run --rm -it --entrypoint /bin/bash --net=host -w /home/zephyruser/ rselagam/zephyr-armfvp:v1
```

In the docker image, run the following to set up environment and executorch module of zephyr:

```
git config --global user.email "ranganath1000@gmail.com" && \
git config --global user.name "BujSet"
git clone https://github.com/BujSet/zephyr.git
cd zephyr/
git switch -c executorch-module-integration origin/executorch-module-integration
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade cmake==3.31.6
python3 -m pip install west torch numpy
python3 -m pip install -r scripts/requirements.txt
cd ../
west init -l zephyr
west config manifest.project-filter -- +executorch
west -v update
source zephyr/zephyr-env.sh
cd ~/modules/lib/executorch
./install_requirements.sh
./examples/arm/setup.sh --i-agree-to-the-contained-eula  --target-toolchain zephyr
source /home/zephyruser/modules/lib/executorch/examples/arm/ethos-u-scratch/setup_path.sh
export PYTHONPATH=/home/zephyruser/modules/lib
```

Validate setup:

```
cd /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world
python validate_setup.py
```

Should produce check marks for everything

Now build and run the whisper encoder application:

```
cd /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world
python build_model.py --pte-file whisper/encoder.pte
west build -p always -b mps3/corstone300/fvp 
FVP_Corstone_SSE-300_Ethos-U55 -a /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/build/zephyr/zephyr.elf -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file='-'  -C cpu0.CFGITCMSZ=15 -C cpu0.CFGDTCMSZ=15 --simlimit 60
```
