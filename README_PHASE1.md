# Phase 1: IID Federated Malicious-Client Detection

This experiment is a software-only, binary `BENIGN` versus `ATTACK` study using the local CICIoT2023 CSV files. It does not require a physical device, blockchain, Hyperledger Fabric, or PyTorch.

## What is compared

- Aggregators: `fedavg`, coordinate-wise `median`, `trimmed_mean`, `multi_krum`, and `fltrust`.
- Attacks: `sign_flip` and `label_flip`.
- Malicious-client rates: 20%, 30%, and 50% of 20 clients.
- Defences: no protection, simulated ABAC only, anomaly detection only, and ABAC plus anomaly detection.

ABAC is a software policy gate in this phase. It rejects a configurable fraction of malicious clients before training, while leaving credential-valid malicious clients to demonstrate why behavioural anomaly detection is still needed. No blockchain claim is made in this phase.

## Metrics

The runner saves three separate result families:

- **Global IDS quality:** accuracy, macro-F1, precision, recall, ROC-AUC, and a binary confusion matrix.
- **Malicious-client detection:** AUROC, precision, recall, F1, FPR, TPR at 1% FPR, and detection delay.
- **System/aggregation:** per-round aggregation time and number of accepted updates.

`Multi-Krum` is recorded as unsupported when its standard condition `n > 2f + 2` does not hold. With 20 clients this excludes the 50% malicious-client case.

## Run locally

```powershell
python -m pip install -r requirements.txt
python run_phase1.py --data-dir CICIoT2023 --profile quick
```

The `quick` profile is intended for a laptop smoke test. Use the default `full` profile in Colab or a machine with sufficient time and memory:

```powershell
python run_phase1.py --data-dir CICIoT2023 --profile full
```

Results are written to `outputs/phase1/` by default. Re-running with the same seed reuses `outputs/phase1/prepared_data.npz`; delete that file to resample source CSVs.

## Colab

Upload or mount the `CICIoT2023` folder and the project files, install `requirements.txt`, then run:

```python
!python run_phase1.py --data-dir /content/CICIoT2023 --profile full
```
