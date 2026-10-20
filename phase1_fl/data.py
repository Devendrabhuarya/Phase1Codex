from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    client_features: list[np.ndarray]
    client_labels: list[np.ndarray]
    root_features: np.ndarray
    root_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    feature_names: list[str]


def _read_sample(data_dir: Path, rows_per_csv: int, seed: int) -> pd.DataFrame:
    csv_files = sorted(data_dir.rglob("*.csv"))
    if len(csv_files) != 63:
        print(f"Warning: expected 63 CSV files, found {len(csv_files)}.")
    frames: list[pd.DataFrame] = []
    for csv_index, csv_path in enumerate(csv_files):
        frame = pd.read_csv(csv_path, nrows=rows_per_csv, low_memory=False)
        if "Label" not in frame.columns:
            raise ValueError(f"{csv_path} has no Label column")
        frames.append(frame)
        print(f"Loaded {csv_index + 1}/{len(csv_files)}: {csv_path.name} ({len(frame):,} rows)")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined


def prepare_iid_data(
    data_dir: str | Path,
    cache_path: str | Path,
    clients: int,
    rows_per_csv: int,
    max_per_class: int,
    root_fraction: float,
    test_fraction: float,
    seed: int,
) -> PreparedData:
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return PreparedData(
            client_features=list(cached["client_features"]),
            client_labels=list(cached["client_labels"]),
            root_features=cached["root_features"],
            root_labels=cached["root_labels"],
            test_features=cached["test_features"],
            test_labels=cached["test_labels"],
            feature_names=list(cached["feature_names"]),
        )

    source = _read_sample(Path(data_dir), rows_per_csv, seed)
    source["binary_label"] = (source["Label"].astype(str).str.upper() != "BENIGN").astype(np.int64)
    benign = source[source["binary_label"] == 0].head(max_per_class)
    attack = source[source["binary_label"] == 1].head(max_per_class)
    balanced = pd.concat([benign, attack], ignore_index=True).sample(frac=1.0, random_state=seed)
    if balanced["binary_label"].nunique() != 2:
        raise ValueError("Both BENIGN and attack samples are required.")

    feature_names = [column for column in source.columns if column not in {"Label", "binary_label"}]
    features = balanced[feature_names].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.median()).fillna(0.0)
    labels = balanced["binary_label"].to_numpy(dtype=np.int64)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features.to_numpy(dtype=np.float64), labels, test_size=test_fraction, stratify=labels, random_state=seed
    )
    client_pool, root_features, client_pool_labels, root_labels = train_test_split(
        train_features,
        train_labels,
        test_size=root_fraction,
        stratify=train_labels,
        random_state=seed,
    )
    scaler = StandardScaler().fit(client_pool)
    client_pool = scaler.transform(client_pool).astype(np.float64)
    root_features = scaler.transform(root_features).astype(np.float64)
    test_features = scaler.transform(test_features).astype(np.float64)

    client_indices = np.arange(len(client_pool))
    client_splits = np.array_split(np.random.default_rng(seed).permutation(client_indices), clients)
    client_features = [client_pool[index] for index in client_splits]
    client_labels = [client_pool_labels[index] for index in client_splits]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        client_features=np.asarray(client_features, dtype=object),
        client_labels=np.asarray(client_labels, dtype=object),
        root_features=root_features,
        root_labels=root_labels,
        test_features=test_features,
        test_labels=test_labels,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    return PreparedData(client_features, client_labels, root_features, root_labels, test_features, test_labels, feature_names)
