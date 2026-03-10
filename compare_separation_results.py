"""
Compare training results with and without separation loss.

This script helps analyze the impact of separation loss by comparing:
1. Training/validation loss curves
2. Detection mAP scores
3. Separation loss values over time
4. Feature connectivity metrics (if available)

Usage:
    python compare_separation_results.py \
        --baseline outputs/baseline \
        --separation outputs/with_separation \
        --output comparison_report.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compare separation loss experiments")
    p.add_argument("--baseline", type=str, required=True,
                   help="Path to baseline experiment directory (no separation loss)")
    p.add_argument("--separation", type=str, required=True,
                   help="Path to separation loss experiment directory")
    p.add_argument("--output", type=str, default="separation_comparison.txt",
                   help="Output report file")
    return p.parse_args()


def load_training_log(log_path: Path) -> Dict[str, List[float]]:
    """Parse training.log and extract loss curves"""
    if not log_path.exists():
        print(f"Warning: Log file not found: {log_path}")
        return {}
    
    metrics = {
        "epoch": [],
        "total_loss": [],
        "loss_det": [],
        "loss_align": [],
        "loss_separation": [],
        "val_loss": [],
    }
    
    with open(log_path) as f:
        for line in f:
            # Parse training loss lines
            if "Epoch" in line and "Loss:" in line:
                # Example: "Epoch 10 [50/782] Loss: 8.3245 loss_det: 7.8123 ..."
                parts = line.split()
                try:
                    epoch_idx = parts.index("Epoch") + 1
                    epoch = int(parts[epoch_idx].split("[")[0])
                    
                    for key in ["Loss:", "loss_det:", "loss_align:", "loss_separation:"]:
                        if key in parts:
                            idx = parts.index(key) + 1
                            value = float(parts[idx])
                            metrics_key = key.rstrip(":").replace("Loss", "total_loss")
                            if metrics_key in metrics:
                                metrics[metrics_key].append(value)
                                if metrics_key == "total_loss":
                                    metrics["epoch"].append(epoch)
                except (ValueError, IndexError):
                    continue
            
            # Parse validation loss lines
            if "Validation Loss:" in line:
                try:
                    val_loss = float(line.split("Validation Loss:")[1].strip().split()[0])
                    metrics["val_loss"].append(val_loss)
                except (ValueError, IndexError):
                    continue
    
    return {k: v for k, v in metrics.items() if len(v) > 0}


def load_config(config_path: Path) -> Dict:
    """Load experiment config.json"""
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return json.load(f)


def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a metric"""
    if len(values) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "final": 0}
    
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "final": values[-1],
    }


def generate_report(baseline_dir: Path, separation_dir: Path, output_path: Path):
    """Generate comparison report"""
    
    print("Loading experiment data...")
    
    # Load configs
    baseline_config = load_config(baseline_dir / "config.json")
    separation_config = load_config(separation_dir / "config.json")
    
    # Load logs
    baseline_metrics = load_training_log(baseline_dir / "training.log")
    separation_metrics = load_training_log(separation_dir / "training.log")
    
    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("Separation Loss Comparison Report")
    lines.append("=" * 80)
    lines.append("")
    
    # Configuration comparison
    lines.append("Configuration")
    lines.append("-" * 80)
    
    if baseline_config and separation_config:
        sep_weight = separation_config.get("training", {}).get("separation_loss_weight", 0.0)
        lines.append(f"Baseline separation weight: 0.0")
        lines.append(f"Experiment separation weight: {sep_weight}")
        lines.append(f"Separation target layer: {separation_config.get('training', {}).get('separation_target_layer', 'N/A')}")
        lines.append(f"Separation valley margin: {separation_config.get('training', {}).get('separation_valley_margin', 'N/A')}")
    else:
        lines.append("(Config files not found)")
    
    lines.append("")
    
    # Loss comparison
    lines.append("Training Loss Statistics")
    lines.append("-" * 80)
    
    for metric_name in ["total_loss", "loss_det", "loss_align"]:
        if metric_name in baseline_metrics and metric_name in separation_metrics:
            baseline_stats = compute_summary_stats(baseline_metrics[metric_name])
            separation_stats = compute_summary_stats(separation_metrics[metric_name])
            
            lines.append(f"\n{metric_name.upper()}:")
            lines.append(f"  Baseline    - Mean: {baseline_stats['mean']:.4f}  Final: {baseline_stats['final']:.4f}")
            lines.append(f"  Separation  - Mean: {separation_stats['mean']:.4f}  Final: {separation_stats['final']:.4f}")
            
            improvement = (baseline_stats['final'] - separation_stats['final']) / baseline_stats['final'] * 100
            lines.append(f"  → Improvement: {improvement:+.2f}% (negative = worse)")
    
    # Separation loss specific
    if "loss_separation" in separation_metrics:
        sep_stats = compute_summary_stats(separation_metrics["loss_separation"])
        lines.append(f"\nSEPARATION LOSS:")
        lines.append(f"  Mean: {sep_stats['mean']:.4f}")
        lines.append(f"  Std:  {sep_stats['std']:.4f}")
        lines.append(f"  Final: {sep_stats['final']:.4f}")
        lines.append(f"  → Trend: {'Decreasing ✓' if sep_stats['final'] < sep_stats['mean'] else 'Stable or increasing'}")
    
    lines.append("")
    
    # Validation loss
    lines.append("Validation Loss")
    lines.append("-" * 80)
    
    if "val_loss" in baseline_metrics and "val_loss" in separation_metrics:
        baseline_val = compute_summary_stats(baseline_metrics["val_loss"])
        separation_val = compute_summary_stats(separation_metrics["val_loss"])
        
        lines.append(f"Baseline    - Final: {baseline_val['final']:.4f}  Min: {baseline_val['min']:.4f}")
        lines.append(f"Separation  - Final: {separation_val['final']:.4f}  Min: {separation_val['min']:.4f}")
        
        improvement = (baseline_val['final'] - separation_val['final']) / baseline_val['final'] * 100
        lines.append(f"→ Improvement: {improvement:+.2f}%")
    else:
        lines.append("(Validation logs not found)")
    
    lines.append("")
    
    # Summary
    lines.append("=" * 80)
    lines.append("Summary")
    lines.append("=" * 80)
    
    if "total_loss" in baseline_metrics and "total_loss" in separation_metrics:
        baseline_final = baseline_metrics["total_loss"][-1]
        separation_final = separation_metrics["total_loss"][-1]
        
        if separation_final < baseline_final * 0.95:
            lines.append("✅ Separation loss shows significant improvement (>5%)")
        elif separation_final < baseline_final * 0.98:
            lines.append("⚠️  Separation loss shows marginal improvement (2-5%)")
        elif separation_final < baseline_final * 1.02:
            lines.append("❌ Separation loss shows no clear benefit (<2% difference)")
        else:
            lines.append("❌ Separation loss appears to hurt performance")
    
    lines.append("")
    lines.append("Recommendations:")
    
    if "loss_separation" in separation_metrics:
        sep_final = separation_metrics["loss_separation"][-1]
        if sep_final > 0.1:
            lines.append("  • Separation loss is still high - consider training longer")
        elif sep_final < 0.001:
            lines.append("  • Separation loss is very low - may be too easy, increase margin or weight")
    
    if "total_loss" in baseline_metrics and "total_loss" in separation_metrics:
        baseline_final = baseline_metrics["total_loss"][-1]
        separation_final = separation_metrics["total_loss"][-1]
        
        if separation_final > baseline_final * 1.05:
            lines.append("  • Performance degraded - reduce separation_loss_weight")
        elif separation_final < baseline_final * 0.98:
            lines.append("  • Good results - consider increasing weight slightly for more separation")
    
    lines.append("")
    lines.append("Next steps:")
    lines.append("  1. Visualize feature maps to confirm spatial separation")
    lines.append("  2. Compute connectivity metrics (merge_rate) on validation set")
    lines.append("  3. Test detection mAP on unseen test data")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    
    print("\n" + report_text)
    print(f"\nReport saved to: {output_path}")


def main():
    args = parse_args()
    
    baseline_dir = Path(args.baseline)
    separation_dir = Path(args.separation)
    output_path = Path(args.output)
    
    # Validate directories
    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        return
    
    if not separation_dir.exists():
        print(f"Error: Separation directory not found: {separation_dir}")
        return
    
    generate_report(baseline_dir, separation_dir, output_path)


if __name__ == "__main__":
    main()
