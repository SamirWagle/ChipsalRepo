#!/usr/bin/env python3
"""
Run All Experiments for CHiPSAL 2026 Paper

This script reproduces all 8 model configurations from Table 4.
Estimated total time: ~20 hours (single RTX 3090)

Usage:
    python scripts/run_all_experiments.py --k_folds 5 --seed 42
    python scripts/run_all_experiments.py --models m7 --quick  # Test single model
"""

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abiral.experiments import run_kfold_experiment
from abiral.utils import set_seed, setup_logging

# Model configurations matching paper Table 4
MODELS = {
    "m1": {"name": "Text-Only (BGE-M3)", "config": "configs/m1_text_only.yaml"},
    "m2": {"name": "Image-Only (Original)", "config": "configs/m2_image_only.yaml"},
    "m3": {"name": "Image-Only (Text-Removed)", "config": "configs/m3_image_text_removed.yaml"},
    "m4": {"name": "Early Fusion (Concat)", "config": "configs/m4_early_fusion.yaml"},
    "m5": {"name": "Late Fusion (Voting)", "config": "configs/m5_late_voting.yaml"},
    "m6": {"name": "Late Fusion (Bagging)", "config": "configs/m6_late_bagging.yaml"},
    "m7": {"name": "Hybrid Fusion (Primary)", "config": "configs/m7_hybrid_fusion.yaml"},
    "m8": {"name": "Hybrid (Text-Removed)", "config": "configs/m8_hybrid_text_removed.yaml"},
}

# Expected results for validation (from paper Table 4)
EXPECTED_RESULTS = {
    "m1": {"f1_macro": 0.6455, "accuracy": 0.7239},
    "m2": {"f1_macro": 0.5952, "accuracy": 0.6791},
    "m3": {"f1_macro": 0.5535, "accuracy": 0.6343},
    "m4": {"f1_macro": 0.6140, "accuracy": 0.6940},
    "m5": {"f1_macro": 0.6511, "accuracy": 0.6940},
    "m6": {"f1_macro": 0.6267, "accuracy": 0.6567},
    "m7": {"f1_macro": 0.6834, "accuracy": 0.7015},
    "m8": {"f1_macro": 0.6436, "accuracy": 0.7015},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run all CHiPSAL experiments")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODELS.keys()) + ["all"],
        help="Models to run (default: all)",
    )
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/paper_results", help="Output directory"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test (1 fold, 5 epochs)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--skip_completed", action="store_true", help="Skip already completed models")
    parser.add_argument("--validate", action="store_true", help="Validate results against paper")
    return parser.parse_args()


def check_prerequisites():
    """Check if all required files and directories exist."""
    required_files = [
        "train/train.csv",
        "train/train_images/",
        "configs/",
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        logging.error(f"Missing required files/directories: {missing}")
        logging.error("Please ensure data is properly set up.")
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Training will be slow on CPU.")
    else:
        logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    return True


def run_single_experiment(
    model_id: str,
    config_path: str,
    k_folds: int,
    seed: int,
    output_dir: str,
    quick: bool = False,
) -> Dict:
    """Run experiment for a single model configuration."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Model {model_id.upper()}: {MODELS[model_id]['name']}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run k-fold experiment
        results = run_kfold_experiment(
            config_path=config_path,
            k_folds=1 if quick else k_folds,
            seed=seed,
            output_dir=f"{output_dir}/{model_id}",
            num_epochs=5 if quick else None,  # Use config default if not quick
        )
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Model {model_id.upper()} Completed!")
        logger.info(f"Time: {elapsed_str}")
        logger.info(f"F1-Macro: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
        logger.info(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        logger.info(f"{'='*80}\n")
        
        results["elapsed_seconds"] = elapsed
        results["elapsed_time"] = elapsed_str
        return results
        
    except Exception as e:
        logger.error(f"Error running {model_id}: {e}", exc_info=True)
        return {"error": str(e), "elapsed_seconds": time.time() - start_time}


def validate_results(results: Dict[str, Dict], tolerance: float = 0.05):
    """Validate results against expected values from paper."""
    
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*80)
    logger.info("VALIDATION: Comparing results to paper Table 4")
    logger.info("="*80 + "\n")
    
    validation_report = []
    
    for model_id, result in results.items():
        if "error" in result:
            continue
        
        expected = EXPECTED_RESULTS.get(model_id, {})
        if not expected:
            continue
        
        f1_diff = abs(result["mean_f1"] - expected["f1_macro"])
        acc_diff = abs(result["mean_accuracy"] - expected["accuracy"])
        
        f1_match = f1_diff <= tolerance
        acc_match = acc_diff <= tolerance
        
        status = "✓ PASS" if (f1_match and acc_match) else "✗ FAIL"
        
        logger.info(f"{model_id.upper()} - {MODELS[model_id]['name']}")
        logger.info(f"  F1-Macro:  {result['mean_f1']:.4f} (expected {expected['f1_macro']:.4f}) "
                   f"[diff: {f1_diff:.4f}] {'✓' if f1_match else '✗'}")
        logger.info(f"  Accuracy:  {result['mean_accuracy']:.4f} (expected {expected['accuracy']:.4f}) "
                   f"[diff: {acc_diff:.4f}] {'✓' if acc_match else '✗'}")
        logger.info(f"  Status: {status}\n")
        
        validation_report.append({
            "model": model_id,
            "f1_obtained": result["mean_f1"],
            "f1_expected": expected["f1_macro"],
            "f1_diff": f1_diff,
            "accuracy_obtained": result["mean_accuracy"],
            "accuracy_expected": expected["accuracy"],
            "accuracy_diff": acc_diff,
            "status": status,
        })
    
    return pd.DataFrame(validation_report)


def save_summary(results: Dict[str, Dict], output_dir: str, validate_df: pd.DataFrame = None):
    """Save experiment summary."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = []
    for model_id in sorted(results.keys()):
        result = results[model_id]
        
        if "error" in result:
            summary_data.append({
                "Model": model_id.upper(),
                "Name": MODELS[model_id]["name"],
                "Status": "FAILED",
                "Error": result["error"],
                "Time": result.get("elapsed_time", "N/A"),
            })
        else:
            summary_data.append({
                "Model": model_id.upper(),
                "Name": MODELS[model_id]["name"],
                "F1-Macro": f"{result['mean_f1']:.4f}",
                "F1-Std": f"{result['std_f1']:.4f}",
                "Accuracy": f"{result['mean_accuracy']:.4f}",
                "Acc-Std": f"{result['std_accuracy']:.4f}",
                "Precision": f"{result['mean_precision']:.4f}",
                "Recall": f"{result['mean_recall']:.4f}",
                "Time": result.get("elapsed_time", "N/A"),
                "Status": "COMPLETED",
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_path = f"{output_dir}/experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\nSummary saved to: {summary_path}")
    
    # Save validation report if available
    if validate_df is not None:
        validate_path = f"{output_dir}/validation_report.csv"
        validate_df.to_csv(validate_path, index=False)
        logging.info(f"Validation report saved to: {validate_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")


def main():
    args = parse_args()
    
    # Setup logging
    log_file = f"{args.output_dir}/experiment_log.txt"
    setup_logging(log_file=log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("CHiPSAL 2026 - Full Experiment Suite")
    logger.info("Reproducing Table 4 from Paper")
    logger.info("="*80)
    logger.info(f"K-Folds: {args.k_folds}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Quick Mode: {args.quick}")
    logger.info("="*80 + "\n")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Determine which models to run
    if args.models is None or "all" in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models
    
    logger.info(f"Running models: {', '.join([m.upper() for m in models_to_run])}\n")
    
    # Run experiments
    all_results = {}
    overall_start = time.time()
    
    for i, model_id in enumerate(models_to_run, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Progress: {i}/{len(models_to_run)} models")
        logger.info(f"{'#'*80}\n")
        
        # Check if already completed
        if args.skip_completed:
            result_file = f"{args.output_dir}/{model_id}/results.json"
            if os.path.exists(result_file):
                logger.info(f"Skipping {model_id} (already completed)")
                continue
        
        # Run experiment
        config_path = MODELS[model_id]["config"]
        results = run_single_experiment(
            model_id=model_id,
            config_path=config_path,
            k_folds=args.k_folds,
            seed=args.seed,
            output_dir=args.output_dir,
            quick=args.quick,
        )
        
        all_results[model_id] = results
    
    # Total time
    total_elapsed = time.time() - overall_start
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info(f"Total Time: {total_elapsed_str}")
    logger.info("="*80 + "\n")
    
    # Validate results if requested
    validate_df = None
    if args.validate and not args.quick:
        validate_df = validate_results(all_results)
    
    # Save summary
    save_summary(all_results, args.output_dir, validate_df)
    
    logger.info(f"\nAll results saved to: {args.output_dir}/")
    logger.info("Experiment suite complete! 🎉\n")


if __name__ == "__main__":
    main()
