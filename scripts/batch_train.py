"""
Batch run experiments sequentially with GPU memory tracking.

Usage:
    python batch_train.py --list              # List experiments
    python batch_train.py 007 008 009         # Run specific experiments
    python batch_train.py --range 007 020     # Run range
    python batch_train.py --pending           # Run all unfinished
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.paths import CHECKPOINTS  # noqa: E402

PROJECT_ROOT = _REPO_ROOT
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# =============================================================================
# GPU MEMORY TRACKING
# =============================================================================

def get_gpu_info():
    """Get GPU memory info or None if CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        
        return {
            "device": torch.cuda.get_device_name(device),
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - reserved,
            "utilization_pct": (reserved / total) * 100 if total > 0 else 0
        }
    except Exception:
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
    except Exception:
        pass
    return False


def print_gpu_status(label=""):
    """Print current GPU memory status bar."""
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
    """Warn if GPU memory is above threshold."""
    info = get_gpu_info()
    if info and info["utilization_pct"] > threshold_pct:
        print(f"\n⚠️  HIGH GPU MEMORY USAGE: {info['utilization_pct']:.0f}%")
        print("   Consider restarting runtime if out-of-memory errors occur.")
        return False
    return True


# =============================================================================
# EXPERIMENT MANAGEMENT
# =============================================================================

def has_trained_model(exp_dir: Path) -> bool:
    """Check if experiment already produced a trained checkpoint."""
    if (exp_dir / "best_model.pt").exists():
        return True
    drive_ckpts = CHECKPOINTS / exp_dir.name
    if drive_ckpts.exists() and any(drive_ckpts.glob("*.pt")):
        return True
    return False


def find_experiment_dir(query: str) -> Path | None:
    """Find experiment directory by 3-digit number or name."""
    pattern = f"{query.zfill(3)}*" if query.isdigit() else f"*{query}*"
    matches = [d for d in EXPERIMENTS_DIR.glob(pattern) if d.is_dir() and not d.name.startswith("_")]
    return matches[0] if matches else None


def list_experiments():
    """List all experiments and their completion status."""
    print("\n📋 Experiments\n")
    print(f"{'#':<5} {'Name':<35} {'Status':<10}")
    print("-" * 55)
    
    pending_count = 0
    done_count = 0
    
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        
        done = has_trained_model(exp_dir)
        status = "✅ Done" if done else "⏳ Pending"
        if done:
            done_count += 1
        else:
            pending_count += 1
        print(f"{exp_dir.name[:3]:<5} {exp_dir.name:<35} {status}")
    
    print(f"\nTotal: {done_count} done, {pending_count} pending")
    print_gpu_status()


def run_single(exp_dir: Path, track_gpu=True) -> dict:
    """Run train.py for a single experiment directory."""
    train_script = exp_dir / "train.py"
    if not train_script.exists():
        print(f"❌ No train.py in {exp_dir.name}")
        return {"success": False, "name": exp_dir.name, "error": "no train.py"}
    
    print(f"\n🚀 Running {exp_dir.name}...")
    print("=" * 60)
    
    gpu_before = print_gpu_status("Before") if track_gpu else None
    start_time = time.time()
    
    result = subprocess.run([sys.executable, str(train_script)], cwd=str(exp_dir))
    
    elapsed = time.time() - start_time
    success = result.returncode == 0
    
    gpu_after = None
    if track_gpu:
        print()
        gpu_after = print_gpu_status("After")
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


def run_batch(exp_dirs: list[Path], track_gpu=True) -> list[dict]:
    """Run a sequence of experiments with GPU health monitoring and summary."""
    if not exp_dirs:
        print("No experiments to run.")
        return []
    
    print(f"\n🚀 Starting batch run for {len(exp_dirs)} experiment(s)...")
    results = []
    
    for exp_dir in exp_dirs:
        if track_gpu and not check_gpu_health(threshold_pct=90):
            try:
                if input("High VRAM usage. Continue anyway? [y/N]: ").strip().lower() != 'y':
                    print("Stopping batch.")
                    break
            except (EOFError, KeyboardInterrupt):
                break
        
        result = run_single(exp_dir, track_gpu=track_gpu)
        results.append(result)
    
    if len(results) > 1:
        succeeded = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        total_time = sum(r.get("duration", 0) for r in results)
        
        print("\n" + "=" * 60)
        print("📊 BATCH SUMMARY")
        print("=" * 60)
        print(f"✅ Succeeded: {len(succeeded)}")
        print(f"❌ Failed:    {len(failed)}")
        print(f"⏱️  Total time: {total_time/60:.1f}m ({total_time/3600:.1f}h)")
        
        if failed:
            print("\nFailed experiments:")
            for r in failed:
                print(f"  - {r.get('name', '?')}: {r.get('error', 'failed with non-zero exit code')}")
        
        print()
        print_gpu_status("Final")
    
    return results


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
        start, end = args.range
        targets = []
        for num in range(start, end + 1):
            d = find_experiment_dir(str(num))
            if d:
                targets.append(d)
            else:
                print(f"⚠️  Experiment {num:03d} not found, skipping")
        run_batch(targets, track_gpu=track_gpu)
    elif args.pending:
        targets = [
            d for d in sorted(EXPERIMENTS_DIR.iterdir())
            if d.is_dir() and not d.name.startswith("_") and (d / "train.py").exists() and not has_trained_model(d)
        ]
        if not targets:
            print("✅ No pending experiments found.")
        else:
            run_batch(targets, track_gpu=track_gpu)
    elif args.experiments:
        targets = []
        for exp_id in args.experiments:
            d = find_experiment_dir(exp_id)
            if d:
                targets.append(d)
            else:
                print(f"❌ Experiment not found: {exp_id}")
        run_batch(targets, track_gpu=track_gpu)
    else:
        list_experiments()


if __name__ == "__main__":
    main()
