.. _executorch_hello_world:

ExecuTorch Hello World
######################

Overview
********

This sample demonstrates how to run ExecuTorch inference on embedded devices
using Zephyr RTOS. It includes a complete build pipeline for model generation,
operator selection, and header file creation. Board-specific configurations 
enable the appropriate backend (ARM, etc.) for your target platform.

Features:
- Embedded PyTorch model (simple addition)
- Selective operator building for reduced memory usage
- Clean program loader implementation
- Comprehensive logging and error handling

Requirements
************

- Python 3.8+ with PyTorch and ExecuTorch
- Zephyr SDK with ARM toolchain
- west build system

Building
********

1. **Enable ExecuTorch Module (First Time Only)**

   Since ExecuTorch is an optional module, enable it first:

   .. code-block:: bash

      west config manifest.project-filter -- +executorch
      west update

2. **Generate Model and Build Files**

   Run the build script to generate all necessary files:

   .. code-block:: bash

      cd zephyr/samples/modules/executorch/hello_world
      python ../scripts/build_model.py

   This script will:
   - Generate ``add.pte`` from the PyTorch model
   - Create ``gen_ops_def.yml`` with required operators
   - Convert the .pte file to ``src/model_pte.h``
   - Make the model data const for flash storage

3. **Build the Zephyr Application**

   For boards with specific configurations (like ARM backend for nRF54L15):

   .. code-block:: bash

      west build -b nrf54l15dk/nrf54l15/cpuapp

   For other boards, you may need to create a board-specific configuration 
   file in the ``boards/`` directory to enable the appropriate backend.

4. **Flash and Run**

   .. code-block:: bash

      west flash

QEMU Emulation
**************

For development and testing without physical hardware, you can run the sample
in QEMU using the Cortex-M3 emulator:

Building for QEMU
==================

Build the sample for the QEMU Cortex-M3 target:

.. code-block:: bash

   west build -p always -b qemu_cortex_m3 samples/modules/executorch/hello_world

The QEMU configuration automatically optimizes memory usage for the limited
64KB RAM environment:

- Method allocator: 4KB (reduced from 16KB)
- Temporary allocator: 512B (reduced from 2KB)
- Reduced stack sizes and disabled optional features

Running in QEMU
================

Execute the built application in QEMU:

.. code-block:: bash

   west build -b qemu_cortex_m3 samples/modules/executorch/hello_world -t run

The application will start automatically. To exit QEMU, press ``Ctrl+A`` then ``X``.

QEMU Memory Usage
=================

The optimized QEMU configuration uses approximately 74% of available RAM:

.. code-block:: text

   Memory region         Used Size  Region Size  %age Used
           FLASH:      166928 B       256 KB     63.68%
             RAM:       48512 B        64 KB     74.02%

Docker Development Environment
******************************

For **native simulation on macOS/Windows**, Zephyr's POSIX architecture requires 
Linux. A Docker environment is provided to solve this limitation.

**Problem**: Zephyr's ``native_sim`` only works on Linux, and the build requires:

- Python 3.12+ (Ubuntu 24.04 or newer)
- Buck2 build system for ExecuTorch 
- Additional Python packages (certifi, etc.)

Setting Up Docker Environment
=============================

1. **Build Minimal Test Environment**:

   .. code-block:: bash

      docker build -f Dockerfile.minimal -t minimal-zephyr-test .

2. **Test Native Simulation Build**:

   .. code-block:: bash

      # For ARM64 macOS (Apple Silicon) - use 64-bit variant
      docker run --rm -v $(pwd):/workspace minimal-zephyr-test bash -c \
        "cd /workspace/zephyr && west build -b native_sim/native/64 samples/modules/executorch/hello_world"

      # For x86_64 systems - use standard variant  
      docker run --rm -v $(pwd):/workspace minimal-zephyr-test bash -c \
        "cd /workspace/zephyr && west build -b native_sim samples/modules/executorch/hello_world"

**Current Status**: The build progresses successfully until ExecuTorch's Buck2 dependency. 
The CMakeLists.txt correctly detects the POSIX architecture and selects the portable backend.

**Next Steps**: Complete Buck2 setup or configure ExecuTorch without Buck2 for native simulation.

Docker Features
===============

- **Ubuntu 24.04**: Python 3.12 support for Zephyr compatibility
- **Minimal Dependencies**: Only essential packages for testing  
- **Automatic Backend Selection**: Uses Portable backend for native, ARM backend for hardware
- **64-bit Support**: Separate configuration for ARM64 and x86_64 architectures

Board Configurations Available:

- ``boards/native_sim.conf`` - 32-bit native simulation (32KB/8KB memory pools)
- ``boards/native_sim_native_64.conf`` - 64-bit native simulation (64KB/16KB memory pools)

Expected Output
***************

The application will load the embedded model, run inference on test data,
and verify the results:

.. code-block:: text

   [00:00:00.123,456] <inf> main: ExecuTorch ARM Hello World Sample
   [00:00:00.234,567] <inf> program_loader: ProgramLoader initialized
   [00:00:00.345,678] <inf> program_loader: Loading embedded program, size: 1234 bytes
   [00:00:00.456,789] <inf> program_loader: Program loaded successfully
   [00:00:00.567,890] <inf> program_loader: Method 'forward' loaded successfully
   [00:00:00.678,901] <inf> main: Program loaded successfully
   [00:00:00.789,012] <inf> program_loader: Running inference with inputs of size 1
   [00:00:00.890,123] <inf> program_loader: Method executed successfully
   [00:00:00.901,234] <inf> program_loader: Inference completed, output size: 1 elements
   [00:00:01.012,345] <inf> main: Inference result: 2.00 + 3.00 = 5.00
   [00:00:01.123,456] <inf> main: ✓ Test PASSED: Addition worked correctly!

Build Pipeline Details
**********************

Model Generation (``build_model.py``)
======================================

The build script automates the complete model pipeline:

1. **Model Export**: Runs ``example_files/export_add.py`` to create ``add.pte``
2. **Operator Analysis**: Uses ExecuTorch's ``gen_ops_def.py`` to determine required operators
3. **Header Generation**: Converts the .pte file to a C header with ``pte_to_header.py``
4. **Const Conversion**: Makes the model data const to store in flash memory

Selective Building
==================

The CMakeLists.txt automatically detects if ``gen_ops_def.yml`` exists and:
- Uses selective operators for minimal memory usage
- Falls back to default portable operators if no selective build file exists
- Links the appropriate operator library

Customization
*************

Creating Your Own Model
=======================

1. Create a new export script in ``example_files/`` (e.g., ``export_mymodel.py``)
2.    Run the build script with your model:

   .. code-block:: bash

      python ../scripts/build_model.py --model-name mymodel

3. Update ``main.cpp`` to use appropriate input/output sizes and data types

Memory Configuration
====================

Adjust memory pool sizes in ``program_loader.cpp``:

.. code-block:: c

   #define ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE (64 * 1024)
   #define ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE (4 * 1024)

Troubleshooting
***************

Common Issues:
- **Build fails**: Ensure ExecuTorch is properly integrated as a Zephyr module
- **Memory errors**: Increase allocator pool sizes
- **Model loading fails**: Check that ``model_pte.h`` is generated correctly
- **Inference fails**: Verify input tensor sizes match model expectations

Architecture
************

The sample consists of:

- ``main.cpp``: Application entry point and test logic
- ``program_loader.h/cpp``: Clean ExecuTorch program management
- ``arm_memory_allocator.hpp``: ARM-optimized memory allocator
- ``build_model.py``: Automated build pipeline
- ``CMakeLists.txt``: Selective operator building support 