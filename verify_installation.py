#!/usr/bin/env python3
"""
Installation verification script for DIVERSED package.
Run this script to check if all dependencies are properly installed.
"""

import sys
import os

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}: OK")
        return True
    except ImportError as e:
        print(f"✗ {description}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠ {description}: WARNING - {e}")
        return False

def test_transformers_version():
    """Test transformers installation and version."""
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        print(f"✓ Transformers location: {transformers.__file__}")
        
        # Test specific imports that are needed
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        print("✓ Required transformers classes: OK")
        return True
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠ Transformers warning: {e}")
        return False

def test_package_structure():
    """Test if package structure is correct."""
    required_dirs = [
        "src",
        "train", 
        "scripts",
        "configs",
        "transformers",
        "utils"
    ]
    
    all_good = True
    for dirname in required_dirs:
        if os.path.exists(dirname) and os.path.isdir(dirname):
            print(f"✓ Directory {dirname}: OK")
        else:
            print(f"✗ Directory {dirname}: MISSING")
            all_good = False
    
    return all_good

def main():
    """Run all verification tests."""
    print("DIVERSED Installation Verification")
    print("=" * 50)
    
    # Test basic Python modules
    print("\n1. Testing basic dependencies:")
    print("-" * 30)
    basic_modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("accelerate", "Accelerate"),
        ("evaluate", "Evaluate"),
        ("sacrebleu", "SacreBLEU"),
    ]
    
    basic_ok = True
    for module, desc in basic_modules:
        if not test_import(module, desc):
            basic_ok = False
    
    # Test transformers specifically
    print("\n2. Testing transformers library:")
    print("-" * 30)
    transformers_ok = test_transformers_version()
    
    # Test package structure
    print("\n3. Testing package structure:")
    print("-" * 30)
    structure_ok = test_package_structure()
    
    # Test package imports
    print("\n4. Testing package imports:")
    print("-" * 30)
    package_imports = [
        ("train.dataloader", "Data loader"),
        ("src.models", "Model definitions"),
        ("utils.rewards", "Reward functions"),
    ]
    
    package_ok = True
    for module, desc in package_imports:
        if not test_import(module, desc):
            package_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Basic dependencies: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"Transformers library: {'✓ PASS' if transformers_ok else '✗ FAIL'}")
    print(f"Package structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Package imports: {'✓ PASS' if package_ok else '✗ FAIL'}")
    
    overall_ok = basic_ok and transformers_ok and structure_ok and package_ok
    print(f"\nOVERALL: {'✓ PASS' if overall_ok else '✗ FAIL'}")
    
    if overall_ok:
        print("\n🎉 Installation verified successfully!")
        print("You can now run the inference examples from the README.")
    else:
        print("\n⚠️ Installation issues detected.")
        print("\nTroubleshooting steps:")
        if not transformers_ok:
            print("1. Install transformers: cd transformers && pip install -e .")
        if not basic_ok:
            print("2. Install basic dependencies: pip install -e .")
        if not package_ok:
            print("3. Make sure you're running from the root directory")
        print("4. See README.md for detailed installation instructions")
    
    return 0 if overall_ok else 1

if __name__ == "__main__":
    sys.exit(main())
