from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve


@dataclass(frozen=True)
class ExperimentSpec:
    aggregator: str
    defence: str
    attack: str
    malicious_fraction: float


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def probabilities(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return sigmoid(features @ weights[:-1] + weights[-1])


def local_update(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, epochs: int, learning_rate: float) -> np.ndarray:
    local_weights = weights.copy()
    for _ in range(epochs):
        error = probabilities(features, local_weights) - labels
        gradient = np.concatenate(((features.T @ error) / len(features), [error.mean()]))
        local_weights -= learning_rate * gradient
    return local_weights - weights


def clip_update(update: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(update)
    return update * min(1.0, clip_norm / (norm + 1e-12))


def coordinate_median(updates: np.ndarray) -> np.ndarray:
    return np.median(updates, axis=0)


def trimmed_mean(updates: np.ndarray, trim_fraction: float) -> np.ndarray:
    trim_count = int(len(updates) * trim_fraction)
    if trim_count == 0 or 2 * trim_count >= len(updates):
        return updates.mean(axis=0)
    ordered = np.sort(updates, axis=0)
    return ordered[trim_count:-trim_count].mean(axis=0)


def multi_krum(updates: np.ndarray, estimated_byzantine: int) -> np.ndarray | None:
    client_count = len(updates)
    if client_count <= 2 * estimated_byzantine + 2:
        return None
    neighbour_count = client_count - estimated_byzantine - 2
    distances = np.sum((updates[:, None, :] - updates[None, :, :]) ** 2, axis=2)
    scores = np.sort(distances, axis=1)[:, 1 : neighbour_count + 1].sum(axis=1)
    selected_count = max(1, client_count - estimated_byzantine)
    return updates[np.argsort(scores)[:selected_count]].mean(axis=0)


def fltrust(updates: np.ndarray, root_update: np.ndarray) -> np.ndarray:
    root_norm = np.linalg.norm(root_update) + 1e-12
    update_norms = np.linalg.norm(updates, axis=1) + 1e-12
    trust = np.maximum(0.0, (updates @ root_update) / (update_norms * root_norm))
    if trust.sum() <= 1e-12:
        return np.zeros_like(root_update)
    normalized = updates * (root_norm / update_norms)[:, None]
    return (normalized * trust[:, None]).sum(axis=0) / trust.sum()


def anomaly_scores(updates: np.ndarray, root_update: np.ndarray) -> np.ndarray:
    reference = coordinate_median(updates)
    reference_norm = np.linalg.norm(reference) + 1e-12
    norms = np.linalg.norm(updates, axis=1)
    cosine_distance = 1.0 - (updates @ reference) / ((norms + 1e-12) * reference_norm)
    median_distance = np.linalg.norm(updates - reference, axis=1)
    root_distance = np.linalg.norm(updates - root_update, axis=1)
    feature_matrix = np.column_stack([norms, cosine_distance, median_distance, root_distance])
    medians = np.median(feature_matrix, axis=0)
    mad = np.median(np.abs(feature_matrix - medians), axis=0) + 1e-9
    robust_z = np.abs(feature_matrix - medians) / (1.4826 * mad)
    return 1.0 - np.exp(-np.mean(robust_z, axis=1))


def abac_admitted(malicious: np.ndarray, block_fraction: float, rng: np.random.Generator) -> np.ndarray:
    admitted = np.ones(len(malicious), dtype=bool)
    attackers = np.flatnonzero(malicious)
    blocked_count = int(round(len(attackers) * block_fraction))
    if blocked_count:
        admitted[rng.choice(attackers, size=blocked_count, replace=False)] = False
    return admitted


def apply_attack(update: np.ndarray, labels: np.ndarray, features: np.ndarray, weights: np.ndarray, attack: str, epochs: int, learning_rate: float) -> np.ndarray:
    if attack == "sign_flip":
        return -5.0 * update
    if attack == "label_flip":
        return local_update(features, 1 - labels, weights, epochs, learning_rate)
    raise ValueError(f"Unsupported attack: {attack}")


def classification_metrics(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    scores = probabilities(features, weights)
    predictions = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tn": int(((predictions == 0) & (labels == 0)).sum()),
        "fp": int(((predictions == 1) & (labels == 0)).sum()),
        "fn": int(((predictions == 0) & (labels == 1)).sum()),
        "tp": int(((predictions == 1) & (labels == 1)).sum()),
    }


def detector_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    if len(np.unique(labels)) < 2:
        return {"detection_auroc": float("nan"), "detection_ap": float("nan"), "detection_precision": float(precision), "detection_recall": float(recall), "detection_f1": float(f1), "detection_fpr": float("nan"), "tpr_at_1pct_fpr": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    valid_tpr = tpr[fpr <= 0.01]
    return {
        "detection_auroc": float(roc_auc_score(labels, scores)),
        "detection_ap": float(average_precision_score(labels, scores)),
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1),
        "detection_fpr": float(((predictions == 1) & (labels == 0)).sum() / max(1, (labels == 0).sum())),
        "tpr_at_1pct_fpr": float(valid_tpr.max() if len(valid_tpr) else 0.0),
    }


def run_experiment(data, spec: ExperimentSpec, rounds: int, local_epochs: int, learning_rate: float, clip_norm: float, anomaly_threshold: float, abac_block_fraction: float, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    client_count = len(data.client_features)
    attacker_count = int(round(client_count * spec.malicious_fraction))
    malicious = np.zeros(client_count, dtype=bool)
    malicious[rng.choice(client_count, size=attacker_count, replace=False)] = True
    abac_mask = abac_admitted(malicious, abac_block_fraction, rng) if "abac" in spec.defence else np.ones(client_count, dtype=bool)
    active_malicious = int((malicious & abac_mask).sum())
    print(
        "Client roles: "
        f"benign={int((~malicious).sum())}, assigned_malicious={attacker_count}, "
        f"acting_malicious={active_malicious}, blocked_malicious={attacker_count - active_malicious}, "
        f"total={client_count}"
    )
    weights = np.zeros(data.client_features[0].shape[1] + 1)
    root_update = local_update(data.root_features, data.root_labels, weights, local_epochs, learning_rate)
    round_rows: list[dict] = []
    first_detected: dict[int, int] = {}
    unsupported = False

    for round_number in range(1, rounds + 1):
        updates = []
        participant_ids = []
        for client_id, (features, labels) in enumerate(zip(data.client_features, data.client_labels)):
            if not abac_mask[client_id]:
                continue
            update = local_update(features, labels, weights, local_epochs, learning_rate)
            if malicious[client_id]:
                update = apply_attack(update, labels, features, weights, spec.attack, local_epochs, learning_rate)
            updates.append(clip_update(update, clip_norm))
            participant_ids.append(client_id)
        updates_array = np.asarray(updates)
        participant_ids_array = np.asarray(participant_ids)
        participant_malicious = malicious[participant_ids_array].astype(int)
        root_update = clip_update(local_update(data.root_features, data.root_labels, weights, local_epochs, learning_rate), clip_norm)
        scores = anomaly_scores(updates_array, root_update)
        accepted = np.ones(len(updates_array), dtype=bool)
        if "anomaly" in spec.defence:
            accepted = scores < anomaly_threshold
            if not accepted.any():
                accepted[np.argmin(scores)] = True
        for client_id, score, is_malicious in zip(participant_ids_array, scores, participant_malicious):
            if is_malicious and score >= anomaly_threshold and client_id not in first_detected:
                first_detected[int(client_id)] = round_number
        usable_updates = updates_array[accepted]
        usable_malicious = participant_malicious[accepted]
        started = perf_counter()
        if spec.aggregator == "fedavg":
            aggregate = usable_updates.mean(axis=0)
        elif spec.aggregator == "median":
            aggregate = coordinate_median(usable_updates)
        elif spec.aggregator == "trimmed_mean":
            aggregate = trimmed_mean(usable_updates, trim_fraction=0.2)
        elif spec.aggregator == "multi_krum":
            aggregate = multi_krum(usable_updates, int(usable_malicious.sum()))
            if aggregate is None:
                unsupported = True
                break
        elif spec.aggregator == "fltrust":
            aggregate = fltrust(usable_updates, root_update)
        else:
            raise ValueError(f"Unknown aggregator {spec.aggregator}")
        aggregation_seconds = perf_counter() - started
        weights += aggregate
        ids = classification_metrics(data.test_features, data.test_labels, weights)
        detection = detector_metrics(scores, participant_malicious, anomaly_threshold)
        round_rows.append({
            "round": round_number,
            "aggregator": spec.aggregator,
            "defence": spec.defence,
            "attack": spec.attack,
            "malicious_fraction": spec.malicious_fraction,
            "accepted_updates": int(accepted.sum()),
            "admitted_malicious": int(participant_malicious.sum()),
            "accepted_malicious": int(usable_malicious.sum()),
            "aggregation_seconds": aggregation_seconds,
            **ids,
            **detection,
        })

    summary = {
        "aggregator": spec.aggregator,
        "defence": spec.defence,
        "attack": spec.attack,
        "malicious_fraction": spec.malicious_fraction,
        "status": "unsupported" if unsupported else "completed",
        "detection_delay": float(np.mean(list(first_detected.values()))) if first_detected else float("nan"),
        "detected_attackers": len(first_detected),
        "total_attackers": attacker_count,
    }
    if round_rows:
        summary.update({key: round_rows[-1][key] for key in ["accuracy", "f1", "roc_auc", "detection_auroc", "detection_f1", "tpr_at_1pct_fpr", "aggregation_seconds", "accepted_updates", "accepted_malicious"]})
    return round_rows, summary
