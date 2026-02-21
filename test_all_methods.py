#!/usr/bin/env python3
"""
Comprehensive test script for DIVERSED methods.
Tests all documented methods to ensure they work correctly.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def run_command(cmd, timeout=120):
    """Run a command with timeout and return success status and output."""
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e)

def test_method(method, draft_model=None, extra_args=None, custom_model_path=None):
    """Test a specific method."""
    print(f"\n{'='*60}")
    print(f"Testing method: {method}")
    print(f"{'='*60}")
    
    # Set model path
    model_path = custom_model_path if custom_model_path else f"./data/cnndm/{method}_test"
    
    # Base command
    cmd = [
        "torchrun", "--nproc_per_node=8", f"--master_port={29500 + hash(method) % 1000}",
        "src/speculative_decoding_dp.py",
        "--method", method,
        "--target_model", "meta-llama/Llama-3.1-8B-Instruct",
        "--dataset", "cnndm",
        "--model_path", model_path,
        "--max_tokens", "16",
        "--n_examples", "1",
        "--do_sample", "False",
        "--temperature", "0.0"
    ]
    
    # Add draft model if needed
    if draft_model:
        cmd.extend(["--draft_model", draft_model])
    
    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    success, stdout, stderr = run_command(cmd, timeout=180)
    
    if success:
        print(f"✅ {method} method: PASSED")
        
        # Check if results file was created
        results_file_path = model_path if custom_model_path else f"./data/cnndm/{method}_test"
        results_file = f"{results_file_path}/{method}_0.5_dynamic_generations.json"
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"   Generated {len(results)} results")
            except:
                print("   Results file exists but couldn't parse JSON")
        else:
            print("   Warning: Results file not found")
            
        return True
    else:
        print(f"❌ {method} method: FAILED")
        if stderr:
            print(f"   Error: {stderr[:500]}...")
        return False

def main():
    """Main test function."""
    print("DIVERSED Methods Test Suite")
    print("=" * 60)
    
    # Ensure we're in the right directory
    if not os.path.exists("src/speculative_decoding_dp.py"):
        print("Error: Must run from diversed_code_release directory")
        sys.exit(1)
    
    # Create data directory if it doesn't exist
    os.makedirs("data/cnndm", exist_ok=True)
    
    # Test results
    results = {}
    
    # Test methods that don't need draft models
    print("\n🔍 Testing methods without draft models...")
    
    # 1. Auto (baseline)
    results["auto"] = test_method("auto")
    
    # Test methods that need draft models
    print("\n🔍 Testing methods with draft models...")
    draft_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    # 2. Standard Speculative Decoding
    results["sd"] = test_method("sd", draft_model)
    
    # 3. Static Ensemble
    results["static_en"] = test_method("static_en", draft_model, ["--draft_ensemble_weights", "0.3"])
    
    # 4. Lossy Speculative Decoding
    results["lossy"] = test_method("lossy", draft_model, ["--lenience", "0.5"])
    
    # 5. Speculative Cascading
    results["spe_cas"] = test_method("spe_cas", draft_model, ["--lenience", "0.5"])
    
    # Test method aliases
    print("\n🔍 Testing method aliases...")
    
    # 6. sd_lossy (alias for lossy)
    results["sd_lossy"] = test_method("sd_lossy", draft_model, ["--lenience", "0.5"])
    
    # 7. sd_static (alias for static_en)
    results["sd_static"] = test_method("sd_static", draft_model, ["--draft_ensemble_weights", "0.3"])
    
    # Test sd_en with sample trained checkpoint
    print("\n🔍 Testing method with trained checkpoint...")
    
    # 8. sd_en (requires trained ensemble head)
    results["sd_en"] = test_method("sd_en", draft_model, extra_args=[], custom_model_path="./data/cnndm/sample_trained_checkpoint")
    
    # Note: diversed is an alias for sd_en
    print("\n⚠️  Note: 'diversed' is an alias for 'sd_en' and uses the same trained model")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for method, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{method:12} : {status}")
    
    print(f"\nOverall: {passed}/{total} methods passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! All documented methods are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} methods failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
