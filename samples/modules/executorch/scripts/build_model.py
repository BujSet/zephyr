#!/usr/bin/env python3
"""
Build script for ExecuTorch samples
Generates .pte model, operator definitions, and header files
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path

def run_command(cmd, cwd=None, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    # Convert Path objects to strings for display
    cmd_str = [str(arg) for arg in cmd]
    print(f"Command: {' '.join(cmd_str)}")
    
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
    parser = argparse.ArgumentParser(description="Generate ExecuTorch headers from existing .pte model")
    parser.add_argument("--executorch-root", required=True,
                       help="Path to ExecuTorch root directory")
    parser.add_argument("--model-file", required=True, 
                       help="Filename of the .pte model (e.g., model_add.pte)")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean generated files before building")
    parser.add_argument("--sample-dir", 
                       help="Sample directory path (auto-detected if not provided)")
    parser.add_argument("--build-dir", 
                       help="Build directory path for generated files")
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).resolve().parent  # Resolve the absolute path first
    # From scripts -> executorch -> modules -> samples -> zephyr -> petriok (5 levels up)
    project_root = script_dir.parent.parent.parent.parent.parent
    
    # Sample directory can be provided or auto-detected
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
        if not sample_dir.is_absolute():
            sample_dir = project_root / sample_dir
    else:
        # Auto-detect: assume we're called from a sample directory
        sample_dir = Path.cwd()
    
    # Build directory for generated files
    if args.build_dir:
        build_dir = Path(args.build_dir)
        if not build_dir.is_absolute():
            build_dir = sample_dir / args.build_dir
    else:
        build_dir = sample_dir / "build"
    
    # Ensure build directory exists
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the provided executorch path
    executorch_root = Path(args.executorch_root)
    if not executorch_root.is_absolute():
        executorch_root = project_root / executorch_root
    
    print(f"Using executorch path: {executorch_root}")
    example_files_dir = project_root / "example_files"
    
    model_file = args.model_file
    pte_file_path = example_files_dir / model_file
    ops_def_file = build_dir / "gen_ops_def.yml"
    header_file = build_dir / "model_pte.h"
    
    print(f"Processing ExecuTorch model: {model_file}")
    print(f"ExecuTorch root: {executorch_root}")
    print(f"Sample directory: {sample_dir}")
    print(f"Build directory: {build_dir}")
    print(f"Working directory: {sample_dir}")
    
    # Check if the .pte file exists
    if not pte_file_path.exists():
        print(f"Error: Model file not found: {pte_file_path}")
        print(f"Please ensure the .pte file is generated manually and placed in {example_files_dir}")
        sys.exit(1)
    
    print(f"✓ Using pre-generated model file: {pte_file_path}")
    
    # Clean previous build if requested
    if args.clean:
        files_to_clean = [ops_def_file, header_file]
        for file_path in files_to_clean:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"Cleaned: {file_path}")
    
    # Step 1: Generate operator definitions
    gen_ops_script = executorch_root / "codegen" / "tools" / "gen_ops_def.py"
    if not gen_ops_script.exists():
        print(f"Error: gen_ops_def.py not found at {gen_ops_script}")
        sys.exit(1)
    
    run_command(
        [sys.executable, str(gen_ops_script), 
         "--output_path", str(ops_def_file),
         "--model_file_path", str(pte_file_path)],
        cwd=sample_dir,
        description="Generating operator definitions"
    )
    
    # Step 2: Convert .pte to header file
    pte_to_header_script = executorch_root / "examples" / "arm" / "executor_runner" / "pte_to_header.py"
    if not pte_to_header_script.exists():
        print(f"Error: pte_to_header.py not found at {pte_to_header_script}")
        sys.exit(1)
    
    run_command(
        [sys.executable, str(pte_to_header_script),
         "--pte", str(pte_file_path),
         "--outdir", str(build_dir)],
        cwd=sample_dir,
        description="Converting .pte to header file"
    )
    
    # Step 3: Make the generated array const and make alignment cross-compiler compatible
    if header_file.exists():
        content = header_file.read_text()
        
        # Make alignment cross-compiler compatible and remove section attributes
        import re
        
        # Add Zephyr headers at the top if not present
        if '#include <zephyr/sys/util.h>' not in content:
            # Insert after any existing includes or at the beginning
            include_pos = content.find('#include')
            if include_pos == -1:
                content = '#include <zephyr/sys/util.h>\n\n' + content
            else:
                # Find the end of the last include
                lines = content.split('\n')
                include_end = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        include_end = i
                lines.insert(include_end + 1, '#include <zephyr/sys/util.h>')
                content = '\n'.join(lines)
        
        # Replace section+aligned pattern with portable alignment
        content = re.sub(r'__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*,\s*aligned\s*\(([^)]*)\)\s*\)\s*\)\s*', 
                        r'__aligned(\1) ', content)
        
        # Remove any remaining section-only attributes  
        content = re.sub(r'__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*\)\s*\)\s*', '', content)
        
        # Replace standalone __attribute__((aligned(n))) with portable __aligned(n)
        content = re.sub(r'__attribute__\s*\(\s*\(\s*aligned\s*\(([^)]*)\)\s*\)\s*\)\s*', 
                        r'__aligned(\1) ', content)
        
        # Replace 'char model_pte_data[]' with 'const char model_pte_data[]'
        content = content.replace('char model_pte_data[]', 'const char model_pte_data[]')
        # Also handle 'char model_pte[]' variant
        content = content.replace('char model_pte[]', 'const char model_pte[]')
        
        header_file.write_text(content)
        print(f"✓ Made model data const and applied portable alignment in {header_file.name}")
    else:
        print(f"Warning: Header file {header_file} not found")
    
    print("\n=== Build Summary ===")
    print(f"✓ Used: {pte_file_path}")
    print(f"✓ Generated: {ops_def_file}")
    print(f"✓ Generated: {header_file}")

if __name__ == "__main__":
    main() 