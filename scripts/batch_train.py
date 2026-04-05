"""
Batch run experiments sequentially with GPU memory tracking.

Usage:
    python batch_train.py --list              # List experiments
    python batch_train.py 007 008 009         # Run specific experiments
    python batch_train.py --range 007 020     # Run range
    python batch_train.py --pending           # Run all unfinished
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# =============================================================================
# GPU MEMORY TRACKING
# =============================================================================

def get_gpu_info():
    """Get GPU memory info. Returns dict with memory stats or None if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        free = total - reserved
        
        return {
            "device": torch.cuda.get_device_name(device),
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "utilization_pct": (reserved / total) * 100
        }
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️  GPU info error: {e}")
        return None


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            return True
    except ImportError:
        pass
    return False


def print_gpu_status(label=""):
    """Print current GPU memory status."""
    info = get_gpu_info()
    if info:
        prefix = f"[{label}] " if label else ""
        bar_len = 20
        used_blocks = int(info["utilization_pct"] / 100 * bar_len)
        bar = "█" * used_blocks + "░" * (bar_len - used_blocks)
        print(f"  {prefix}GPU: [{bar}] {info['reserved_gb']:.1f}/{info['total_gb']:.1f}GB ({info['utilization_pct']:.0f}%)")
        return info
    return None


def check_gpu_health(threshold_pct=80):
    """Check if GPU memory is below threshold. Warn if high."""
    info = get_gpu_info()
    if info and info["utilization_pct"] > threshold_pct:
        print(f"\n⚠️  HIGH GPU MEMORY USAGE: {info['utilization_pct']:.0f}%")
        print(f"   Consider restarting the runtime to free memory.")
        return False
    return True


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def list_experiments():
    """List all experiments and their status."""
    print("\n📋 Experiments\n")
    print(f"{'#':5} {'Name':35} {'Status':10}")
    print("-" * 55)
    
    pending_count = 0
    done_count = 0
    
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        
        # Check if has trained model
        has_model = (exp_dir / "best_model.pt").exists() or \
                   any((PROJECT_ROOT.parent.parent.parent / "models/trained" / exp_dir.name).glob("*.pt"))
        
        if has_model:
            status = "✅ Done"
            done_count += 1
        else:
            status = "⏳ Pending"
            pending_count += 1
        print(f"{exp_dir.name[:3]:5} {exp_dir.name:35} {status}")
    
    print(f"\nTotal: {done_count} done, {pending_count} pending")
    
    # Show GPU info
    print_gpu_status()


def run_experiment(exp_name_or_num: str, track_gpu=True):
    """Run a single experiment with optional GPU tracking."""
    # Find experiment directory
    if exp_name_or_num.isdigit():
        matches = list(EXPERIMENTS_DIR.glob(f"{exp_name_or_num.zfill(3)}_*"))
    else:
        matches = list(EXPERIMENTS_DIR.glob(f"*{exp_name_or_num}*"))
    
    if not matches:
        print(f"❌ Experiment not found: {exp_name_or_num}")
        return {"success": False, "error": "not found"}
    
    exp_dir = matches[0]
    train_script = exp_dir / "train.py"
    
    if not train_script.exists():
        print(f"❌ No train.py in {exp_dir.name}")
        return {"success": False, "error": "no train.py"}
    
    print(f"\n🚀 Running {exp_dir.name}...")
    print("=" * 60)
    
    # GPU status before
    gpu_before = None
    if track_gpu:
        gpu_before = print_gpu_status("Before")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(exp_dir),
    )
    
    elapsed = time.time() - start_time
    success = result.returncode == 0
    
    # GPU status after
    gpu_after = None
    if track_gpu:
        print()
        gpu_after = print_gpu_status("After")
        
        # Clear GPU memory between experiments
        if clear_gpu_memory():
            print("  🧹 GPU memory cache cleared")
    
    print(f"\n{'✅' if success else '❌'} {exp_dir.name}: {'completed' if success else 'failed'} in {elapsed/60:.1f}m")
    
    return {
        "success": success,
        "name": exp_dir.name,
        "duration": elapsed,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after
    }


def run_range(start: int, end: int, track_gpu=True):
    """Run experiments in a range."""
    print(f"\n🚀 Running experiments {start:03d} to {end:03d}...")
    
    results = []
    for num in range(start, end + 1):
        matches = list(EXPERIMENTS_DIR.glob(f"{num:03d}_*"))
        if matches:
            # Check GPU health before each run
            if track_gpu and not check_gpu_health(threshold_pct=90):
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Stopping batch.")
                    break
            
            result = run_experiment(str(num), track_gpu=track_gpu)
            results.append(result)
    
    _print_batch_summary(results)
    return results


def run_pending(track_gpu=True):
    """Run all experiments without trained models."""
    pending = []
    
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        if not (exp_dir / "train.py").exists():
            continue
        
        # Check if already trained
        has_model = any(Path(PROJECT_ROOT.parent.parent.parent / "models/trained" / exp_dir.name).glob("*.pt"))
        if not has_model:
            pending.append(exp_dir)
    
    if not pending:
        print("✅ No pending experiments!")
        return []
    
    print(f"\n🚀 Running {len(pending)} pending experiments...")
    
    results = []
    for exp_dir in pending:
        # Check GPU health before each run
        if track_gpu and not check_gpu_health(threshold_pct=90):
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("Stopping batch.")
                break
        
        result = run_experiment(exp_dir.name[:3], track_gpu=track_gpu)
        results.append(result)
    
    _print_batch_summary(results)
    return results


def run_multiple(experiments, track_gpu=True):
    """Run multiple specified experiments."""
    results = []
    for exp in experiments:
        if track_gpu and not check_gpu_health(threshold_pct=90):
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("Stopping batch.")
                break
        
        result = run_experiment(exp, track_gpu=track_gpu)
        results.append(result)
    
    if len(results) > 1:
        _print_batch_summary(results)
    return results


def _print_batch_summary(results):
    """Print summary of batch training run."""
    if not results:
        return
    
    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    total_time = sum(r.get("duration", 0) for r in results)
    
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Succeeded: {len(succeeded)}")
    print(f"❌ Failed:    {len(failed)}")
    print(f"⏱️  Total time: {total_time/60:.1f}m ({total_time/3600:.1f}h)")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r.get('name', '?')}: {r.get('error', 'unknown error')}")
    
    # Final GPU status
    print()
    print_gpu_status("Final")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch run experiments with GPU tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_train.py --list                # List all experiments
  python batch_train.py 007                   # Run experiment 007
  python batch_train.py 007 008 009           # Run multiple experiments
  python batch_train.py --range 007 020       # Run experiments 007-020
  python batch_train.py --pending             # Run all unfinished
  python batch_train.py --pending --no-gpu    # Without GPU tracking
        """
    )
    
    parser.add_argument("experiments", nargs="*", help="Experiment numbers or names to run")
    parser.add_argument("--list", "-l", action="store_true", help="List all experiments")
    parser.add_argument("--range", "-r", nargs=2, type=int, metavar=("START", "END"),
                       help="Run experiments in range")
    parser.add_argument("--pending", "--new", action="store_true", 
                       help="Run all pending (unfinished) experiments")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU memory tracking")
    
    args = parser.parse_args()
    track_gpu = not args.no_gpu
    
    if args.list:
        list_experiments()
    elif args.range:
        run_range(args.range[0], args.range[1], track_gpu=track_gpu)
    elif args.pending:
        run_pending(track_gpu=track_gpu)
    elif args.experiments:
        run_multiple(args.experiments, track_gpu=track_gpu)
    else:
        list_experiments()


if __name__ == "__main__":
    main()
