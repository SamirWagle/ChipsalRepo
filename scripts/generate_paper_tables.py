#!/usr/bin/env python3
"""
Generate LaTeX tables from experimental results.

This script reads result CSV files and produces LaTeX tables matching
the paper format (Table 4: Model comparison).

Usage:
    python generate_paper_tables.py --results results.csv --output tables.tex
    python generate_paper_tables.py --results results.csv --format markdown
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> pd.DataFrame:
    """Load experimental results from CSV."""
    logger.info(f"Loading results from {results_path}")
    df = pd.DataFrame(pd.read_csv(results_path))
    logger.info(f"Loaded {len(df)} experimental results")
    return df


def generate_latex_table(df: pd.DataFrame, caption: str = "") -> str:
    """
    Generate LaTeX table from results DataFrame.
    
    Expected columns: model, f1_macro, precision, recall, accuracy
    """
    latex = []
    
    # Table header
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{" + caption + "}")
    latex.append("\\label{tab:results}")
    latex.append("\\begin{tabular}{l|cccc}")
    latex.append("\\hline")
    latex.append("\\textbf{Model} & \\textbf{F1-Macro} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{Accuracy} \\\\")
    latex.append("\\hline")
    
    # Add rows
    for _, row in df.iterrows():
        model = row["model"]
        f1 = row["f1_macro"]
        prec = row.get("precision", 0.0)
        rec = row.get("recall", 0.0)
        acc = row.get("accuracy", 0.0)
        
        # Highlight best model (highest F1)
        if f1 == df["f1_macro"].max():
            latex.append(f"\\textbf{{{model}}} & \\textbf{{{f1:.4f}}} & {prec:.4f} & {rec:.4f} & {acc:.4f} \\\\")
        else:
            latex.append(f"{model} & {f1:.4f} & {prec:.4f} & {rec:.4f} & {acc:.4f} \\\\")
    
    # Table footer
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate Markdown table from results DataFrame."""
    md = []
    
    # Header
    md.append("| Model | F1-Macro | Precision | Recall | Accuracy |")
    md.append("|-------|----------|-----------|--------|----------|")
    
    # Rows
    for _, row in df.iterrows():
        model = row["model"]
        f1 = row["f1_macro"]
        prec = row.get("precision", 0.0)
        rec = row.get("recall", 0.0)
        acc = row.get("accuracy", 0.0)
        
        # Highlight best with bold
        if f1 == df["f1_macro"].max():
            md.append(f"| **{model}** | **{f1:.4f}** | {prec:.4f} | {rec:.4f} | {acc:.4f} |")
        else:
            md.append(f"| {model} | {f1:.4f} | {prec:.4f} | {rec:.4f} | {acc:.4f} |")
    
    return "\n".join(md)


def generate_ablation_table(df: pd.DataFrame) -> str:
    """
    Generate ablation study table comparing model variants.
    
    Groups models by fusion strategy.
    """
    latex = []
    
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation Study: Impact of Fusion Strategies}")
    latex.append("\\label{tab:ablation}")
    latex.append("\\begin{tabular}{lc|c}")
    latex.append("\\hline")
    latex.append("\\textbf{Category} & \\textbf{Model} & \\textbf{F1-Macro} \\\\")
    latex.append("\\hline")
    
    # Baselines
    latex.append("\\multirow{3}{*}{Baselines}")
    for model_name in ["M1: Text-Only", "M2: Image-Only", "M3: Image (Text-Removed)"]:
        row = df[df["model"] == model_name]
        if not row.empty:
            f1 = row.iloc[0]["f1_macro"]
            latex.append(f" & {model_name} & {f1:.4f} \\\\")
    
    latex.append("\\hline")
    
    # Fusion methods
    latex.append("\\multirow{3}{*}{Fusion}")
    for model_name in ["M4: Early Fusion", "M5: Late Voting", "M6: Late Bagging"]:
        row = df[df["model"] == model_name]
        if not row.empty:
            f1 = row.iloc[0]["f1_macro"]
            latex.append(f" & {model_name} & {f1:.4f} \\\\")
    
    latex.append("\\hline")
    
    # Hybrid
    latex.append("\\multirow{2}{*}{Hybrid}")
    for model_name in ["M7: Hybrid Fusion", "M8: Hybrid (Text-Removed)"]:
        row = df[df["model"] == model_name]
        if not row.empty:
            f1 = row.iloc[0]["f1_macro"]
            # Bold best model
            if model_name == "M7: Hybrid Fusion":
                latex.append(f" & \\textbf{{{model_name}}} & \\textbf{{{f1:.4f}}} \\\\")
            else:
                latex.append(f" & {model_name} & {f1:.4f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_confusion_matrix_latex(confusion: Dict[str, List[int]]) -> str:
    """Generate LaTeX confusion matrix."""
    latex = []
    
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Confusion Matrix: M7 Hybrid Fusion}")
    latex.append("\\label{tab:confusion}")
    latex.append("\\begin{tabular}{cc|cc}")
    latex.append("\\hline")
    latex.append("& & \\multicolumn{2}{c}{\\textbf{Predicted}} \\\\")
    latex.append("& & Hate & Non-Hate \\\\")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Actual}}")
    
    # Get confusion matrix values
    tn, fp = confusion.get("non_hate", [0, 0])
    fn, tp = confusion.get("hate", [0, 0])
    
    latex.append(f" & Hate & {tp} & {fn} \\\\")
    latex.append(f" & Non-Hate & {fp} & {tn} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experimental results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tables.tex",
        help="Output file path (default: tables.tex)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["latex", "markdown"],
        default="latex",
        help="Output format (default: latex)"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Generate ablation study table"
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Model Comparison: F1-Macro Scores",
        help="Table caption"
    )
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results)
    
    # Generate table
    if args.format == "latex":
        if args.ablation:
            table = generate_ablation_table(df)
        else:
            table = generate_latex_table(df, caption=args.caption)
    else:
        table = generate_markdown_table(df)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table)
    
    logger.info(f"Generated {args.format} table: {output_path}")
    
    # Print to console
    print("\n" + "="*80)
    print(f"Generated {args.format.upper()} Table:")
    print("="*80)
    print(table)
    print("="*80)
    
    logger.info("✓ Table generation complete")


if __name__ == "__main__":
    main()
