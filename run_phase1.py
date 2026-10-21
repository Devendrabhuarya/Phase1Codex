from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_fl.data import prepare_iid_data
from phase1_fl.federated import ExperimentSpec, run_experiment
from phase1_fl.visualize import save_plots


def log_traffic_class_counts(data) -> None:
    """Print the binary traffic-label distribution across all prepared splits."""
    labels = np.concatenate([
        *data.client_labels,
        data.root_labels,
        data.test_labels,
    ])
    benign_count = int((labels == 0).sum())
    attack_count = int((labels == 1).sum())
    print(
        "Traffic samples: "
        f"benign={benign_count:,}, attack={attack_count:,}, total={len(labels):,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 IID FL malicious-client detection experiment")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if not arguments.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {arguments.data_dir}")

    configuration = {
        "quick": {"rows_per_csv": 1_000, "max_per_class": 15_000, "rounds": 5, "local_epochs": 1},
        "full": {"rows_per_csv": 5_000, "max_per_class": 100_000, "rounds": 15, "local_epochs": 2},
    }[arguments.profile]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_iid_data(
        data_dir=arguments.data_dir,
        cache_path=arguments.output_dir / "prepared_data.npz",
        clients=20,
        rows_per_csv=configuration["rows_per_csv"],
        max_per_class=configuration["max_per_class"],
        root_fraction=0.02,
        test_fraction=0.20,
        seed=arguments.seed,
    )
    log_traffic_class_counts(data)
    specifications = [
        ExperimentSpec(aggregator, defence, attack, malicious_fraction)
        for aggregator in ["fedavg", "median", "trimmed_mean", "multi_krum", "fltrust"]
        for defence in ["none", "abac", "anomaly", "abac_anomaly"]
        for attack in ["sign_flip", "label_flip"]
        for malicious_fraction in [0.20, 0.30, 0.50]
    ]
    all_rounds: list[dict] = []
    summaries: list[dict] = []
    for index, specification in enumerate(specifications, start=1):
        print(f"[{index}/{len(specifications)}] {specification}")
        rounds, summary = run_experiment(
            data=data,
            spec=specification,
            rounds=configuration["rounds"],
            local_epochs=configuration["local_epochs"],
            learning_rate=0.15,
            clip_norm=5.0,
            anomaly_threshold=0.72,
            abac_block_fraction=0.50,
            seed=arguments.seed + index,
        )
        all_rounds.extend(rounds)
        summaries.append(summary)
    rounds_frame = pd.DataFrame(all_rounds)
    summary_frame = pd.DataFrame(summaries)
    rounds_frame.to_csv(arguments.output_dir / "round_metrics.csv", index=False)
    summary_frame.to_csv(arguments.output_dir / "summary_metrics.csv", index=False)
    save_plots(rounds_frame, summary_frame, arguments.output_dir / "plots")
    print(f"Saved results to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
