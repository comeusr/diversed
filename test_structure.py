#!/usr/bin/env python3
"""
Simple test to verify the package structure without requiring heavy dependencies.
"""

import os
import sys
import importlib.util

def test_file_exists(filepath, description):
    """Test if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def test_python_syntax(filepath, description):
    """Test if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print(f"✓ {description}: Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {description}: Syntax error - {e}")
        return False
    except Exception as e:
        print(f"✗ {description}: Error reading file - {e}")
        return False

def main():
    """Run structure tests."""
    print("DIVERSED Package Structure Test")
    print("=" * 50)
    
    # Test essential files
    essential_files = [
        ("README.md", "README file"),
        ("setup.py", "Setup script"),
        ("requirements.txt", "Requirements file"),
        ("LICENSE", "License file"),
        ("src/__init__.py", "Source package init"),
        ("src/models.py", "Model definitions"),
        ("src/speculative_decoding.py", "Speculative decoding implementation"),
        ("src/speculative_decoding_dp.py", "Data parallel speculative decoding"),
        ("train/__init__.py", "Training package init"),
        ("train/trainer.py", "Trainer implementation"),
        ("train/dataloader.py", "Data loader implementation"),
        ("scripts/__init__.py", "Scripts package init"),
        ("scripts/run_dp.py", "Data parallel inference script"),
        ("scripts/run_inference.py", "General inference script"),
        ("scripts/run_train.py", "Training script"),
        ("configs/default_config.yaml", "Default configuration"),
        ("configs/template.jinja", "Chat template"),
        ("utils/__init__.py", "Utils package init"),
        ("utils/rewards.py", "Reward functions"),
    ]
    
    print("\n1. Testing file existence:")
    print("-" * 30)
    all_files_exist = True
    for filepath, description in essential_files:
        if not test_file_exists(filepath, description):
            all_files_exist = False
    
    # Test Python syntax for key files
    python_files = [
        ("src/models.py", "Model definitions"),
        ("src/speculative_decoding.py", "Speculative decoding"),
        ("train/trainer.py", "Trainer"),
        ("scripts/run_dp.py", "Data parallel script"),
        ("scripts/run_inference.py", "Inference script"),
    ]
    
    print("\n2. Testing Python syntax:")
    print("-" * 30)
    all_syntax_valid = True
    for filepath, description in python_files:
        if os.path.exists(filepath):
            if not test_python_syntax(filepath, description):
                all_syntax_valid = False
        else:
            print(f"⚠ {description}: File not found, skipping syntax check")
    
    # Test directory structure
    print("\n3. Testing directory structure:")
    print("-" * 30)
    required_dirs = [
        "src/mydatasets",
        "data",
        "logs",
        "transformers",
    ]
    
    all_dirs_exist = True
    for dirname in required_dirs:
        if os.path.exists(dirname) and os.path.isdir(dirname):
            print(f"✓ Directory exists: {dirname}")
        else:
            print(f"✗ Directory missing: {dirname}")
            all_dirs_exist = False
    
    # Test dataset-specific files
    print("\n4. Testing dataset files:")
    print("-" * 30)
    dataset_files = [
        "src/mydatasets/humaneval/evaluate.py",
        "src/mydatasets/mbpp/evaluate.py",
        "src/mydatasets/xsum/prompt_zeroshot.txt",
        "src/mydatasets/wmt/prompt_zeroshot.txt",
    ]
    
    dataset_files_exist = True
    for filepath in dataset_files:
        if os.path.exists(filepath):
            print(f"✓ Dataset file exists: {filepath}")
        else:
            print(f"✗ Dataset file missing: {filepath}")
            dataset_files_exist = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Essential files: {'✓ PASS' if all_files_exist else '✗ FAIL'}")
    print(f"Python syntax: {'✓ PASS' if all_syntax_valid else '✗ FAIL'}")
    print(f"Directory structure: {'✓ PASS' if all_dirs_exist else '✗ FAIL'}")
    print(f"Dataset files: {'✓ PASS' if dataset_files_exist else '✗ FAIL'}")
    
    overall_pass = all_files_exist and all_syntax_valid and all_dirs_exist and dataset_files_exist
    print(f"\nOVERALL: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    if overall_pass:
        print("\n🎉 Package structure looks good!")
        print("Next steps:")
        print("1. Install dependencies: pip install -e .")
        print("2. Install transformers: cd transformers && pip install -e .")
        print("3. Run inference examples from the README")
    else:
        print("\n⚠️  Some issues found. Please check the failed items above.")
    
    return 0 if overall_pass else 1

if __name__ == "__main__":
    sys.exit(main())
