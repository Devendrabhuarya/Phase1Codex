from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_plots(rounds: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    completed = summary[summary["status"] == "completed"].copy()
    for metric, title in [("accuracy", "Final global-model accuracy"), ("f1", "Final global-model F1"), ("detection_auroc", "Malicious-client detection AUROC")]:
        figure, axis = plt.subplots(figsize=(13, 6))
        sns.barplot(data=completed, x="aggregator", y=metric, hue="defence", errorbar=None, ax=axis)
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}_comparison.png", dpi=180)
        plt.close(figure)
    if not rounds.empty:
        figure, axis = plt.subplots(figsize=(13, 6))
        sns.lineplot(data=rounds, x="round", y="accuracy", hue="aggregator", style="defence", errorbar=None, ax=axis)
        axis.set_title("Global-model accuracy by federated round")
        axis.set_ylim(0, 1.05)
        figure.tight_layout()
        figure.savefig(output_dir / "accuracy_by_round.png", dpi=180)
        plt.close(figure)

        for group_key, group in rounds.groupby(["aggregator", "defence", "attack", "malicious_fraction"]):
            final = group.sort_values("round").iloc[-1]
            matrix = [[final["tn"], final["fp"]], [final["fn"], final["tp"]]]
            figure, axis = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"], ax=axis)
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("True label")
            axis.set_title(f"{group_key[0]} | {group_key[1]} | {group_key[2]} | {group_key[3]:.0%}")
            figure.tight_layout()
            safe_name = "_".join(str(item).replace(".", "_") for item in group_key)
            figure.savefig(output_dir / f"confusion_{safe_name}.png", dpi=180)
            plt.close(figure)
