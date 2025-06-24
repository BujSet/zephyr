#!/usr/bin/env python3
"""
Validation script for ExecuTorch ARM Hello World setup
Checks that all required files are present and properly configured
"""

import os
import sys
from pathlib import Path

def check_file(file_path, description):
    """Check if a file exists and print status"""
    if file_path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False

def check_file_content(file_path, search_text, description):
    """Check if a file contains specific content"""
    if not file_path.exists():
        print(f"❌ {description}: {file_path} (FILE NOT FOUND)")
        return False
    
    try:
        content = file_path.read_text()
        if search_text in content:
            print(f"✅ {description}: {file_path}")
            return True
        else:
            print(f"❌ {description}: {file_path} (CONTENT NOT FOUND)")
            return False
    except Exception as e:
        print(f"❌ {description}: {file_path} (ERROR: {e})")
        return False

def main():
    print("🔍 Validating ExecuTorch ARM Hello World Setup")
    print("=" * 50)
    
    script_dir = Path(__file__).parent
    src_dir = script_dir / "src"
    
    all_good = True
    
    # Check main files
    files_to_check = [
        (script_dir / "CMakeLists.txt", "CMake build file"),
        (script_dir / "prj.conf", "Project configuration"),
        (script_dir / "README.rst", "Documentation"),
        (script_dir / "build_model.py", "Build pipeline script"),
        (src_dir / "main.cpp", "Main application"),
        (src_dir / "program_loader.h", "Program loader header"),
        (src_dir / "program_loader.cpp", "Program loader implementation"),
        (src_dir / "arm_memory_allocator.hpp", "ARM memory allocator"),
        (src_dir / "model_pte.h", "Model header")
    ]
    
    for file_path, description in files_to_check:
        if not check_file(file_path, description):
            all_good = False
    
    print("\n📋 Checking File Contents")
    print("-" * 30)
    
    # Check specific content
    content_checks = [
        (src_dir / "program_loader.h", "class ProgramLoader", "ProgramLoader class definition"),
        (src_dir / "program_loader.cpp", "ProgramLoader::", "ProgramLoader implementation"),
        (src_dir / "main.cpp", "ProgramLoader::getInstance", "Main app uses ProgramLoader"),
        (src_dir / "model_pte.h", "char model_pte[]", "Model data array"),
        (script_dir / "CMakeLists.txt", "gen_selected_ops", "Selective operator building"),
        (script_dir / "build_model.py", "gen_ops_def.py", "Operator definition generation")
    ]
    
    for file_path, search_text, description in content_checks:
        if not check_file_content(file_path, search_text, description):
            all_good = False
    
    print("\n🔧 Build Pipeline Test")
    print("-" * 20)
    
    # Test build script
    build_script = script_dir / "build_model.py"
    if build_script.exists() and os.access(build_script, os.X_OK):
        print("✅ Build script is executable")
    else:
        print("❌ Build script is not executable")
        all_good = False
    
    print("\n📦 Required Dependencies")
    print("-" * 25)
    
    try:
        import torch
        print("✅ PyTorch is available")
        torch_available = True
    except ImportError:
        print("⚠️  PyTorch not available (needed for model generation)")
        torch_available = False
    
    # Check if ExecuTorch paths exist
    project_root = script_dir.parent.parent.parent.parent.parent.parent
    executorch_root = project_root / "modules" / "lib" / "executorch"
    
    if executorch_root.exists():
        print(f"✅ ExecuTorch found at: {executorch_root}")
        
        # Check for specific ExecuTorch tools
        gen_ops_script = executorch_root / "codegen" / "tools" / "gen_ops_def.py"
        pte_to_header_script = executorch_root / "examples" / "arm" / "executor_runner" / "pte_to_header.py"
        
        if gen_ops_script.exists():
            print("✅ gen_ops_def.py tool found")
        else:
            print("❌ gen_ops_def.py tool not found")
            all_good = False
            
        if pte_to_header_script.exists():
            print("✅ pte_to_header.py tool found")
        else:
            print("❌ pte_to_header.py tool not found")
            all_good = False
    else:
        print(f"❌ ExecuTorch not found at: {executorch_root}")
        all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 SETUP VALIDATION PASSED!")
        print("\nNext steps:")
        if torch_available:
            print("1. Run: python build_model.py")
            print("2. Run: west build -b nrf54l15dk/nrf54l15/cpuapp")
        else:
            print("1. Install PyTorch: pip install torch")
            print("2. Install ExecuTorch (see documentation)")
            print("3. Run: python build_model.py")
            print("4. Run: west build -b nrf54l15dk/nrf54l15/cpuapp")
        print("5. Run: west flash")
    else:
        print("❌ SETUP VALIDATION FAILED!")
        print("Please check the errors above and fix them.")
        sys.exit(1)

if __name__ == "__main__":
    main() 