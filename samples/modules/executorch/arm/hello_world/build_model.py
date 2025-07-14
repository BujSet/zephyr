#!/usr/bin/env python3
"""
Build script for ExecuTorch ARM Hello World sample
Generates .pte model, operator definitions, and header files
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Build ExecuTorch ARM Hello World model")
    parser.add_argument("--executorch-root", default="~/modules/lib/executorch", 
                       help="Path to ExecuTorch root directory")
    parser.add_argument("--pte-file", default="add.pte", 
                       help="Exported model (default: add.pte)")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean generated files before building")
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent.parent  # Go up to petriok root
    executorch_root = args.executorch_root
    sys.path.append(Path(executorch_root).parent)
    example_files_dir = "/home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/example_files"
    src_dir = script_dir / "src"
    
    pte_file = args.pte_file 
    ops_def_file = "gen_ops_def.yml"
    header_file = "model_pte.h"
    
    print(f"ExecuTorch root: {executorch_root}")
    print(f"Working directory: {script_dir}")
    
    # Clean previous build if requested
    if args.clean:
        files_to_clean = [pte_file, ops_def_file, src_dir / header_file]
        for file_path in files_to_clean:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"Cleaned: {file_path}")
    
    # Step 3: Convert .pte to header file
    #pte_to_header_script = executorch_root / "examples" / "arm" / "executor_runner" / "pte_to_header.py"
    pte_to_header_script = "/home/zephyruser/modules/lib/executorch/examples/arm/executor_runner/pte_to_header.py"
    if not os.path.exists(pte_to_header_script):
        print(f"Error: pte_to_header.py not found at {pte_to_header_script}")
        sys.exit(1)
    
    run_command(
        [sys.executable, str(pte_to_header_script),
         "--pte", pte_file,
         "--outdir", "src"],
        cwd=script_dir,
        description="Converting .pte to header file"
    )
    
    # Step 4: Make the generated array const and remove section attribute
    # TODO the header should have the following sugnature
    # __attribute__((aligned(16))) const char model_pte[] = {


    header_path = src_dir / header_file
#    if header_path.exists():
#        content = header_path.read_text()
#        
#        # Remove section attribute and replace with Zephyr alignment macro
#        import re
#        # Replace section+aligned pattern with Zephyr __ALIGN macro
#        content = re.sub(r'__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*,\s*aligned\s*\(([^)]*)\)\s*\)\s*\)\s*', r'__ALIGN(\1) ', content)
#        # Remove any remaining section-only attributes  
#        content = re.sub(r'__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*\)\s*\)\s*', '', content)
#        # Also replace any standalone __attribute__((aligned(n))) with __ALIGN(n)
#        content = re.sub(r'__attribute__\s*\(\s*\(\s*aligned\s*\(([^)]*)\)\s*\)\s*\)\s*', r'__ALIGN(\1) ', content)
#        
#        # Replace 'char model_pte_data[]' with 'const char model_pte_data[]'
#        content = content.replace('char model_pte_data[]', 'const char model_pte_data[]')
#        # Also handle 'char model_pte[]' variant
#        content = content.replace('char model_pte[]', 'const char model_pte[]')
#        
#        header_path.write_text(content)
#        print(f"✓ Made model data const and removed section attributes in {header_file}")
#    else:
#        print(f"Warning: Header file {header_file} not found")
    
    print("\n=== Build Summary ===")
    print(f"✓ Generated: {pte_file}")
    print(f"✓ Generated: {ops_def_file}")
    print(f"✓ Generated: src/{header_file}")
    print("\nNext steps:")
    print("1. Review gen_ops_def.yml and customize if needed")
    print("2. Build the Zephyr application with west build")

if __name__ == "__main__":
    main() 
