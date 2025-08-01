# Copyright (c) 2020-2025 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

# This file includes extra build system logic that is enabled when
# CONFIG_BOOTLOADER_MCUBOOT=y.
#
# It builds signed binaries using imgtool as a post-processing step
# after zephyr/zephyr.elf is created in the build directory.
#
# Since this file is brought in via include(), we do the work in a
# function to avoid polluting the top-level scope.

function(zephyr_runner_file type path)
  # Property magic which makes west flash choose the signed build
  # output of a given type.
  set_target_properties(runners_yaml_props_target PROPERTIES "${type}_file" "${path}")
endfunction()

function(zephyr_mcuboot_tasks)
  set(keyfile "${CONFIG_MCUBOOT_SIGNATURE_KEY_FILE}")
  set(keyfile_enc "${CONFIG_MCUBOOT_ENCRYPTION_KEY_FILE}")
  string(CONFIGURE "${keyfile}" keyfile)
  string(CONFIGURE "${keyfile_enc}" keyfile_enc)

  if(NOT "${CONFIG_MCUBOOT_GENERATE_UNSIGNED_IMAGE}")
    # Check for misconfiguration.
    if("${keyfile}" STREQUAL "")
      # No signature key file, no signed binaries. No error, though:
      # this is the documented behavior.
      message(WARNING "Neither CONFIG_MCUBOOT_GENERATE_UNSIGNED_IMAGE or "
                      "CONFIG_MCUBOOT_SIGNATURE_KEY_FILE are set, the generated build will not be "
                      "bootable by MCUboot unless it is signed manually/externally.")
      return()
    elseif(NOT (CONFIG_BUILD_OUTPUT_BIN OR CONFIG_BUILD_OUTPUT_HEX))
      message(FATAL_ERROR "Can't sign images for MCUboot: Neither "
                          "CONFIG_BUILD_OUTPUT_BIN nor CONFIG_BUILD_OUTPUT_HEX "
                          "is enabled, so there's nothing to sign.")
    endif()

    foreach(file keyfile keyfile_enc)
      if("${${file}}" STREQUAL "")
        continue()
      endif()

      # Find the key files in the order of preference for a simple search
      # modeled by the if checks across the various locations
      #
      #  1. absolute
      #  2. application config
      #  3. west topdir (optional when the workspace is not west managed)
      #
      if(NOT IS_ABSOLUTE "${${file}}")
        if(EXISTS "${APPLICATION_CONFIG_DIR}/${${file}}")
          set(${file} "${APPLICATION_CONFIG_DIR}/${${file}}")
        else()
          # Relative paths are relative to 'west topdir'.
          #
          # This is the only file that has a relative check to topdir likely
          # from the historical callouts to "west" itself before using
          # imgtool. So, this is maintained here for backward compatibility
          #
          if(NOT WEST OR NOT WEST_TOPDIR)
            message(FATAL_ERROR "Can't sign images for MCUboot: west workspace undefined. "
                                "To fix, ensure `west topdir` is a valid workspace directory.")
          endif()
          set(${file} "${WEST_TOPDIR}/${${file}}")
        endif()
      endif()

      if(NOT EXISTS "${${file}}")
        message(FATAL_ERROR "Can't sign images for MCUboot: can't find file ${${file}} "
                            "(Note: Relative paths are searched through "
                            "APPLICATION_CONFIG_DIR=\"${APPLICATION_CONFIG_DIR}\" "
                            "and WEST_TOPDIR=\"${WEST_TOPDIR}\")")
      endif()
    endforeach()
  endif()

  # No imgtool, no signed binaries.
  if(NOT DEFINED IMGTOOL)
    message(FATAL_ERROR "Can't sign images for MCUboot: can't find imgtool. To fix, install imgtool with pip3, or add the mcuboot repository to the west manifest and ensure it has a scripts/imgtool.py file.")
    return()
  endif()

  # Fetch devicetree details for flash and slot information
  dt_chosen(flash_node PROPERTY "zephyr,flash")
  dt_nodelabel(slot0_flash NODELABEL "slot0_partition" REQUIRED)
  dt_prop(slot_size PATH "${slot0_flash}" PROPERTY "reg" INDEX 1 REQUIRED)

  # If single slot mode, or if in firmware updater mode and this is the firmware updater image,
  # use slot 0 information
  if(NOT CONFIG_MCUBOOT_BOOTLOADER_MODE_SINGLE_APP AND (NOT CONFIG_MCUBOOT_BOOTLOADER_MODE_FIRMWARE_UPDATER OR CONFIG_MCUBOOT_APPLICATION_FIRMWARE_UPDATER)
      AND NOT CONFIG_MCUBOOT_BOOTLOADER_MODE_SINGLE_APP_RAM_LOAD)
    # Slot 1 size is used instead of slot 0 size
    set(slot_size)
    dt_nodelabel(slot1_flash NODELABEL "slot1_partition" REQUIRED)
    dt_prop(slot_size PATH "${slot1_flash}" PROPERTY "reg" INDEX 1 REQUIRED)
  endif()

  # Basic 'imgtool sign' command with known image information.
  set(imgtool_sign ${PYTHON_EXECUTABLE} ${IMGTOOL} sign
      --version ${CONFIG_MCUBOOT_IMGTOOL_SIGN_VERSION} --header-size ${CONFIG_ROM_START_OFFSET}
      --slot-size ${slot_size})

  # Arguments to imgtool.
  if(NOT CONFIG_MCUBOOT_EXTRA_IMGTOOL_ARGS STREQUAL "")
    # Separate extra arguments into the proper format for adding to
    # extra_post_build_commands.
    #
    # Use UNIX_COMMAND syntax for uniform results across host
    # platforms.
    separate_arguments(imgtool_args UNIX_COMMAND ${CONFIG_MCUBOOT_EXTRA_IMGTOOL_ARGS})
  else()
    set(imgtool_args)
  endif()

  if(NOT "${keyfile}" STREQUAL "")
    set(imgtool_args --key "${keyfile}" ${imgtool_args})
  endif()

  if(CONFIG_MCUBOOT_IMGTOOL_OVERWRITE_ONLY)
    # Use overwrite-only instead of swap upgrades.
    set(imgtool_args --overwrite-only --align 1 ${imgtool_args})
  elseif(CONFIG_MCUBOOT_BOOTLOADER_MODE_RAM_LOAD)
    # RAM load requires setting the location of where to load the image to
    dt_chosen(chosen_ram PROPERTY "zephyr,sram")
    dt_reg_addr(chosen_ram_address PATH ${chosen_ram})
    dt_nodelabel(slot0_partition NODELABEL "slot0_partition" REQUIRED)
    dt_reg_addr(slot0_partition_address PATH ${slot0_partition})
    dt_nodelabel(slot1_partition NODELABEL "slot1_partition" REQUIRED)
    dt_reg_addr(slot1_partition_address PATH ${slot1_partition})

    set(imgtool_args --align 1 --load-addr ${chosen_ram_address} ${imgtool_args})
    set(imgtool_args_alt_slot ${imgtool_args} --hex-addr ${slot1_partition_address})
    set(imgtool_args ${imgtool_args} --hex-addr ${slot0_partition_address})
  elseif(CONFIG_MCUBOOT_BOOTLOADER_MODE_SINGLE_APP_RAM_LOAD)
    dt_chosen(ram_load_dev PROPERTY "mcuboot,ram-load-dev")
    if(DEFINED ram_load_dev)
      dt_reg_addr(load_address PATH ${ram_load_dev})
    else()
      dt_chosen(chosen_ram PROPERTY "zephyr,sram")
      dt_reg_addr(load_address PATH ${chosen_ram})
    endif()
    set(imgtool_args --align 1 --load-addr ${load_address} ${imgtool_args})
  else()
    dt_prop(write_block_size PATH "${flash_node}" PROPERTY "write-block-size")

    if(NOT write_block_size)
      set(write_block_size 4)
      message(WARNING "slot0_partition write block size devicetree parameter is missing, assuming write block size is 4")
    endif()

    set(imgtool_args --align ${write_block_size} ${imgtool_args})
  endif()

  # Set proper hash calculation algorithm for signing
  if(CONFIG_MCUBOOT_BOOTLOADER_SIGNATURE_TYPE_PURE)
    set(imgtool_args --pure ${imgtool_args})
  elseif(CONFIG_MCUBOOT_BOOTLOADER_USES_SHA512)
    set(imgtool_args --sha 512 ${imgtool_args})
  endif()

  # Extensionless prefix of any output file.
  set(output ${ZEPHYR_BINARY_DIR}/${KERNEL_NAME})

  # List of additional build byproducts.
  set(byproducts)

  # Set up .bin outputs.
  if(CONFIG_BUILD_OUTPUT_BIN)
    list(APPEND byproducts ${output}.signed.bin)
    zephyr_runner_file(bin ${output}.signed.bin)
    set(BYPRODUCT_KERNEL_SIGNED_BIN_NAME "${output}.signed.bin"
        CACHE FILEPATH "Signed kernel bin file" FORCE
    )
    set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                 ${imgtool_sign} ${imgtool_args} ${output}.bin ${output}.signed.bin)

    if(CONFIG_MCUBOOT_GENERATE_CONFIRMED_IMAGE)
      list(APPEND byproducts ${output}.signed.confirmed.bin)
      zephyr_runner_file(bin ${output}.signed.confirmed.bin)
      set(BYPRODUCT_KERNEL_SIGNED_CONFIRMED_BIN_NAME "${output}.signed.confirmed.bin"
          CACHE FILEPATH "Signed and confirmed kernel bin file" FORCE
      )
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args} --pad --confirm ${output}.bin
                   ${output}.signed.confirmed.bin)
    endif()

    if(NOT "${keyfile_enc}" STREQUAL "")
      list(APPEND byproducts ${output}.signed.encrypted.bin)
      set(BYPRODUCT_KERNEL_SIGNED_ENCRYPTED_BIN_NAME "${output}.signed.encrypted.bin"
          CACHE FILEPATH "Signed and encrypted kernel bin file" FORCE
      )
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args} --encrypt "${keyfile_enc}" ${output}.bin
                   ${output}.signed.encrypted.bin)
    endif()

    if(CONFIG_MCUBOOT_BOOTLOADER_MODE_RAM_LOAD)
      list(APPEND byproducts ${output}.slot1.signed.encrypted.bin)
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args_alt_slot} ${output}.bin
                   ${output}.slot1.signed.bin)

      if(CONFIG_MCUBOOT_GENERATE_CONFIRMED_IMAGE)
        list(APPEND byproducts ${output}.slot1.signed.confirmed.bin)
        set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                     ${imgtool_sign} ${imgtool_args_alt_slot} --pad --confirm ${output}.bin
                     ${output}.slot1.signed.confirmed.bin)
      endif()

      if(NOT "${keyfile_enc}" STREQUAL "")
        list(APPEND byproducts ${output}.slot1.signed.encrypted.bin)
        set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                     ${imgtool_sign} ${imgtool_args_alt_slot} --encrypt "${keyfile_enc}"
                     ${output}.bin ${output}.slot1.signed.encrypted.bin)
      endif()
    endif()
  endif()

  # Set up .hex outputs.
  if(CONFIG_BUILD_OUTPUT_HEX)
    list(APPEND byproducts ${output}.signed.hex)
    zephyr_runner_file(hex ${output}.signed.hex)
    set(BYPRODUCT_KERNEL_SIGNED_HEX_NAME "${output}.signed.hex"
        CACHE FILEPATH "Signed kernel hex file" FORCE
    )

    if(NOT "${keyfile_enc}" STREQUAL "")
      # When encryption is enabled, set the encrypted bit when signing the image but do not
      # encrypt the data, this means that when the image is moved out of the primary into the
      # secondary, it will be encrypted rather than being in unencrypted
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args} --encrypt "${keyfile_enc}" --clear
                   ${output}.hex ${output}.signed.hex)
    else()
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args} ${output}.hex ${output}.signed.hex)
    endif()

    if(CONFIG_MCUBOOT_GENERATE_CONFIRMED_IMAGE)
      list(APPEND byproducts ${output}.signed.confirmed.hex)
      zephyr_runner_file(hex ${output}.signed.confirmed.hex)
      set(BYPRODUCT_KERNEL_SIGNED_CONFIRMED_HEX_NAME "${output}.signed.confirmed.hex"
          CACHE FILEPATH "Signed and confirmed kernel hex file" FORCE
      )
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args} --pad --confirm ${output}.hex
                   ${output}.signed.confirmed.hex)
    endif()

    if(NOT "${keyfile_enc}" STREQUAL "")
      list(APPEND byproducts ${output}.signed.encrypted.hex)
      set(BYPRODUCT_KERNEL_SIGNED_ENCRYPTED_HEX_NAME "${output}.signed.encrypted.hex"
          CACHE FILEPATH "Signed and encrypted kernel hex file" FORCE
      )
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args} --encrypt "${keyfile_enc}" ${output}.hex
                   ${output}.signed.encrypted.hex)
    endif()

    if(CONFIG_MCUBOOT_BOOTLOADER_MODE_RAM_LOAD)
      list(APPEND byproducts ${output}.slot1.signed.hex)
      set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                   ${imgtool_sign} ${imgtool_args_alt_slot} ${output}.hex
                   ${output}.slot1.signed.hex)

      if(CONFIG_MCUBOOT_GENERATE_CONFIRMED_IMAGE)
        list(APPEND byproducts ${output}.slot1.signed.confirmed.hex)
        set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                     ${imgtool_sign} ${imgtool_args_alt_slot} --pad --confirm ${output}.hex
                     ${output}.slot1.signed.confirmed.hex)
      endif()

      if(NOT "${keyfile_enc}" STREQUAL "")
        list(APPEND byproducts ${output}.slot1.signed.encrypted.hex)
        set_property(GLOBAL APPEND PROPERTY extra_post_build_commands COMMAND
                     ${imgtool_sign} ${imgtool_args_alt_slot} --encrypt "${keyfile_enc}"
                     ${output}.hex ${output}.slot1.signed.encrypted.hex)
      endif()
    endif()
  endif()
  set_property(GLOBAL APPEND PROPERTY extra_post_build_byproducts ${byproducts})
endfunction()

zephyr_mcuboot_tasks()
