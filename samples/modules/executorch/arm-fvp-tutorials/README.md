# Arm FVP Tutorial

This tutorial is to demonstrate how to run an exported ExecuTorch model on an Arm device running Zephyr via the the Arm FVP simulator.

# Requirements

Requires docker to be installed. While the CLI is sufficient to complete this tutorial, Docker Desktop offers a lot of useful features. Instructions for downloading and installing Docker and/or Docker Desktop can be found [here](https://docs.docker.com/desktop/). You should enable docker to utilizes ~60 GB of VM memory, enabling as much swap that fits within your system. This tutorial was tested and validated on a machine with 34.4GB RAM and 1TB of disk where docker VM was permitted 60GB VM memory and 100GB swap space.

# Quick Start Commands

<details>

You can copy-paste the commands below to setup up and validate a working flow. Start the docker image with the following command:

```
docker run -it --entrypoint /bin/bash --net=host -w /home/zephyruser/ rselagam/zephyr-armfvp:v8
```

Once in the docker image, run the following commands to set up environment and executorch module of zephyr. Please note that you should replace `<YOUR EMAL>` and `<YOUR USER NAME>` with appropriate GitHub credentials:

```
git config --global user.email "<YOUR EMAIL>" && \
git config --global user.name "<YOUR USER NAME>"
git clone https://github.com/BujSet/zephyr.git
cd zephyr/
git switch -c executorch-module-integration origin/executorch-module-integration
cd ../
west init -l zephyr
west config manifest.project-filter -- +executorch
west -v update
cd ~/modules/lib/executorch
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
./install_requirements.sh
./examples/arm/setup.sh --i-agree-to-the-contained-eula  --target-toolchain zephyr
source /home/zephyruser/modules/lib/executorch/examples/arm/ethos-u-scratch/setup_path.sh
python -m examples.arm.aot_arm_compiler --model_name="add" --output="add.pte"
cd ~/zephyr
python3 -m pip install west
python3 -m pip install -r scripts/requirements.txt
source zephyr-env.sh
cd /home/zephyruser/zephyr/samples/modules/executorch/arm-fvp-tutorials/
python3 build_model.py --pte-file=/home/zephyruser/modules/lib/executorch/add.pte --output-path=/home/zephyruser/zephyr/samples/modules/executorch/arm-fvp-tutorials/models/add/src/
cd models/add/
west build -p always -b mps3/corstone300/fvp -- -DET_PTE_FILE_PATH_FOR_SELECTIVE_BUILD=/home/zephyruser/modules/lib/executorch/add.pte && FVP_Corstone_SSE-300_Ethos-U55 -a build/zephyr/zephyr.elf -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file='-'  -C cpu0.CFGITCMSZ=15 -C cpu0.CFGDTCMSZ=15 --simlimit 60
```
</details>

# Normal Setup

Its strongly recommended to use the docker image paired with this tutorial. You can do this by either pulling from Dockerhub (preferred) or building the image from source ([Adding Packages to the Docker Image and Building from Source](#adding-packages-to-the-docker-image-and-building-from-source) section below).

## Pulling the container

```
docker pull rselagam/zephyr-armfvp:v8
```

## Starting the docker image


It's often advantageous to map a shared volume between the hsot machine and the running docker image. This allows a user to transfer files between the two seamlessly. In the commands below `workspace` from the current direct is mapped to `/workspace` within the docker container. This flag option can be omitted if file sharing is not needed.

### Linux/macOS

```
docker run -it --entrypoint /bin/bash  --net=host -v "$(pwd)"/workspace:/workspace -w /home/zephyruser/ rselagam/zephyr-armfvp:v8
```

### Windows (PowerShell)

```
docker run -it --entrypoint /bin/bash --net=host -v "${PWD}\workspace:/workspace" -w /home/zephyruser/ rselagam/zephyr-armfvp:v8
```

# Working inside the Docker image

## Setup (One time)

### Setup Zephyr Repo

```
git clone https://github.com/BujSet/zephyr.git
cd zephyr/
git switch -c executorch-module-integration origin/executorch-module-integration
cd ../
west init -l zephyr
west config manifest.project-filter -- +executorch
west -v update
```

### Setup ExecuTorch

```
cd ~/modules/lib/executorch
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
./install_executorch.sh
```

### Setup FVP Simulator

```
cd ~/modules/lib/executorch/
./examples/arm/setup.sh --i-agree-to-the-contained-eula  --target-toolchain zephyr
source /home/zephyruser/modules/lib/executorch/examples/arm/ethos-u-scratch/setup_path.sh
```


### Setup West Build System

```
cd ~/zephyr
python3 -m pip install west
python3 -m pip install -r scripts/requirements.txt
source zephyr-env.sh
```

## Building an Example

### Create Example PTE

```
cd ~/modules/lib/executorch/
python -m examples.arm.aot_arm_compiler --model_name="add" --output="add.pte"
```

Currently, we support `add`, `softmax`, and `mv2` models.


### Create C-Style Header from PTE

```
cd /home/zephyruser/zephyr/samples/modules/executorch/arm-fvp-tutorials/
python3 build_model.py --pte-file=/home/zephyruser/modules/lib/executorch/add.pte --output-path=/home/zephyruser/zephyr/samples/modules/executorch/arm-fvp-tutorials/models/add/src/
```

After running the commands above, you should see a file called `model_pte.h` created at `/home/zephyruser/zephyr/samples/modules/executorch/arm-fvp-tutorials/models/add/src/`.

### Building the ELF Using Dtype Selective Build

```
cd /home/zephyruser/zephyr/samples/modules/executorch/arm-fvp-tutorials/models/add/
west build -p always -b mps3/corstone300/fvp -- -DET_PTE_FILE_PATH_FOR_SELECTIVE_BUILD=/home/zephyruser/modules/lib/executorch/add.pte
```

And you should see output similar to this:

```
Memory region         Used Size  Region Size  %age Used
           FLASH:      311592 B       512 KB     59.43%
             RAM:       80792 B         1 MB      7.70%
            ITCM:          0 GB       512 KB      0.00%
            SRAM:          0 GB         1 MB      0.00%
            DTCM:          0 GB       512 KB      0.00%
           ISRAM:          0 GB         2 MB      0.00%
          DDR4S0:         944 B       256 MB      0.00%
          DDR4S1:         512 B       256 MB      0.00%
          DDR4S2:       98944 B       256 MB      0.04%
          DDR4S3:          0 GB       256 MB      0.00%
 NULL_PTR_DETECT:          0 GB         1 KB      0.00%
            DDR4:          0 GB       256 MB      0.00%
        IDT_LIST:          0 GB        32 KB      0.00%
```

## Running a model in the simulator

```
FVP_Corstone_SSE-300_Ethos-U55 -a /home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/build/zephyr/zephyr.elf -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file='-' --simlimit 30
```

And you should see output like:

```
telnetterminal0: Listening for serial connection on port 5000
telnetterminal1: Listening for serial connection on port 5001
telnetterminal2: Listening for serial connection on port 5002
telnetterminal5: Listening for serial connection on port 5003

    Ethos-U rev 136b7d75 --- Apr 12 2023 13:44:01
    (C) COPYRIGHT 2019-2023 Arm Limited
    ALL RIGHTS RESERVED

I [executorch:arm_executor_runner.cpp:211 main()] PTE at 0x70000000 Size: 944 bytes
I [executorch:arm_executor_runner.cpp:219 main()] PTE Model data loaded. Size: 944 bytes.
I [executorch:arm_executor_runner.cpp:232 main()] Model buffer loaded, has 1 methods
I [executorch:arm_executor_runner.cpp:240 main()] Running method forward
xterm: xterm: Xt error: Can't open display:
Xt error: Can't open display:
xterm: DISPLAY is not set
xterm: DISPLAY is not set
I [executorch:arm_executor_runner.cpp:251 main()] Setup Method allocator pool. Size: 512 bytes.
I [executorch:arm_executor_runner.cpp:268 main()] Setting up planned buffer 0, size 64.
I [executorch:arm_executor_runner.cpp:283 main()] Computed planned buffer size=64
I [executorch:arm_executor_runner.cpp:298 main()] Loading method.
I [executorch:arm_executor_runner.cpp:311 main()] Method 'forward' loaded.
I [executorch:arm_executor_runner.cpp:314 main()] Preparing input: In use: 320B, Free: 192B
I [executorch:arm_executor_runner.cpp:379 main()] Input prepared.
I [executorch:arm_executor_runner.cpp:381 main()] Starting the model execution...
I [executorch:arm_executor_runner.cpp:387 main()] model_pte_program_size:     944 bytes.
I [executorch:arm_executor_runner.cpp:388 main()] model_pte_loaded_size:      944 bytes.
I [executorch:arm_executor_runner.cpp:391 main()] method_allocator_used:     344 / 512  free: 168 ( used: 67 % )
I [executorch:arm_executor_runner.cpp:398 main()] method_allocator_planned:  64 bytes
I [executorch:arm_executor_runner.cpp:400 main()] method_allocator_loaded:   256 bytes
I [executorch:arm_executor_runner.cpp:401 main()] method_allocator_input:    24 bytes
I [executorch:arm_executor_runner.cpp:402 main()] method_allocator_executor: 0 bytes
I [executorch:arm_executor_runner.cpp:405 main()] peak_temp_allocator:       0 / 2048 free: 2048 ( used: 0 % )
I [executorch:arm_executor_runner.cpp:421 main()] Model executed successfully.
I [executorch:arm_executor_runner.cpp:429 main()] Beginning output verificaiton
I [executorch:arm_executor_runner.cpp:450 main()] SUCCESS: Program complete, exiting.
I [executorch:arm_executor_runner.cpp:451 main()] ♦
*** Booting Zephyr OS build 19451fec644e ***

Info: Simulation is stopping. Reason: Simulated time has been exceeded.

Info: /OSCI/SystemC: Simulation stopped by user.
[warning ][main@0][01 ns] Simulation stopped by user
```

# Troubleshooting

## Adding Packages to the Docker Image and Building from Source

<details>
If the default docker image does not contain all the software packages you desire, you can modify the [Dockerfile.armfvp_zephyr](Dockerfile.armfvp_zephyr) file to include the missing packages.

To build the new image (or if pulling the image from DockerHub failed), you can run the following:

```
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx create --name mybuilder --driver docker-container --platform linux/amd64,linux/arm6 --use
docker buildx build --network=host --platform linux/amd64,linux/arm64 -t my_new_docker_image -f Dockerfile.armfvp_zephyr  .
```

The steps before the last command ensure that the image you build supports deployment to multiple platforms. If you are only interested in build for your local machine, you can run:

```
docker build -t my_new_docker_image -f Dockerfile.armfvp_zephyr  .
```
</details>
