.. _executorch_hello_world:

ExecuTorch ARM Hello World
##########################

Overview
********

This sample demonstrates how to run ExecuTorch inference on ARM Cortex-M devices
using Zephyr RTOS. It includes a complete build pipeline for model generation,
operator selection, and header file creation.

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

      cd zephyr/samples/modules/executorch/arm/hello_world
      python build_model.py

   This script will:

   - Generate ``add.pte`` from the PyTorch model

   - Create ``gen_ops_def.yml`` with required operators

   - Convert the .pte file to ``src/model_pte.h``

   - Make the model data const for flash storage


3. **Build the Zephyr Application**

   .. code-block:: bash

      west build -b nrf54l15dk/nrf54l15/cpuapp
      west build -p always -b mps3/corstone300/an547

4. **Flash and Run**

   .. code-block:: bash

      west flash

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
2. Run the build script with your model:

   .. code-block:: bash

      python build_model.py --model-name mymodel

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
