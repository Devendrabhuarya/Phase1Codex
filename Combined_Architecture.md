# Combined Architecture: ABAC-Secured, Byzantine-Resilient Federated Edge IDS

## 1. Purpose and research boundary

This architecture combines:

1. **Rahman, Wang, and Wei (IEEE Access, 2025)**: Hyperledger Fabric, X.509 identities, and attribute-based access control (ABAC) to prevent unauthorised or stale devices from joining federated learning (FL).
2. **Nayak and Reddy (Discover Artificial Intelligence, 2026)**: a compact edge intrusion-detection model and Byzantine-resilient FL aggregation under non-IID client data.
3. **This project**: cohort-conditioned malicious-client detection, cohort-capture resistance, and a reputation lifecycle for *credential-valid but malicious* clients.

The system trains a global intrusion-detection system (IDS) from local IoT traffic without moving raw flows from devices. It assumes that ABAC stops unauthorised participation but does not assume that an authorised client is honest. Updates from admitted clients are therefore scored before they are aggregated and each client’s history affects future participation.

Two outcomes must always be reported separately:

| Outcome | Question | Example measures |
|---|---|---|
| IDS quality | Does the global model classify network traffic correctly? | macro-F1, balanced accuracy, per-class recall, ROC-AUC |
| Client-security quality | Does the system find model-poisoning clients without harming honest clients? | detection AUROC, TPR at 1% FPR, false revocations, detection delay, attacker weight retained |

Blockchain overhead and access-control performance are third, separate outcomes: transaction latency, throughput, policy-decision latency, and ledger storage per round.

## 2. Scope, data, and trust assumptions

### 2.1 Data and client heterogeneity

- **Primary traffic dataset:** CICIoT2023, with 39 flow features and labels merged into eight classes: `Benign`, `DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, and `BruteForce`.
- **Synthetic device context:** the dataset has no genuine device, session, timestamp, or capture identifiers. Each FL client is assigned a synthetic `device_type`; the parameter `rho ∈ [0,1]` controls how strongly that type determines its class mixture. This must be described as a simulation variable, not a recovered dataset attribute.
- **Non-IID split:** within the device-type constraint, use a reproducible Dirichlet or equivalent class-partition procedure. Report `rho`, the Dirichlet concentration, client sample sizes, and client participation rate.
- **Optional edge model:** use the compact CNN–BiLSTM family from the Byzantine-resilient IDS paper when edge deployment is in scope. It operates on the flow-feature vector and is suitable for constrained devices; a larger compatible model may run on gateways.

### 2.2 Threat model

| Threat | Example | Primary control |
|---|---|---|
| Unauthorised or stale participant | spoofed identity, expired certificate, invalid attribute set | Fabric CA, X.509 validation, ABAC chaincode |
| Authorised Byzantine participant | sign-flipping, label-flipping, model replacement, backdoor update | update anomaly score, robust aggregation, reputation |
| Coordinated cohort capture | attackers dominate one device-type cohort and shift its local reference | global-vs-cohort consistency flag and per-cohort aggregation cap |
| Replay/tampering | reused round update or altered submitted update | round binding, signature, update hash, Fabric audit record |

The FL coordinator/aggregator and Fabric ordering/endorsement infrastructure are trusted for the experiment. Fabric provides identity, policy enforcement, and tamper-evident records; it does not inspect the private update vector or prove that an admitted device trained honestly.

## 3. Logical architecture

```text
                         ┌───────────────────────────────────────────────┐
                         │ Hyperledger Fabric permissioned trust plane    │
                         │ CA → X.509 attributes → ABAC chaincode → log │
                         └───────────────┬───────────────────────────────┘
                                         │ admits / denies
┌──────────────────────┐                 │               ┌─────────────────────────────┐
│ IoT FL clients       │◄──── global W_t ┼──────────────►│ Off-chain FL coordinator    │
│ local CICIoT data    │                 │               │ cohort scoring + aggregation │
│ local CNN–BiLSTM     │── signed Δ_k ───┘               └──────────────┬──────────────┘
└──────────────────────┘                                                 │
                                                                         │ round summary only
                         ┌───────────────────────────────────────────────▼┐
                         │ Fabric world state: reputation, status, hashes, │
                         │ cohort metadata, thresholds, deny-record root   │
                         └─────────────────────────────────────────────────┘
```

The system has three planes:

1. **Device/data plane:** devices retain raw traffic and run local training.
2. **Federated intelligence plane:** an off-chain coordinator validates submitted metadata, forms cohorts, scores updates, aggregates accepted contributions, and updates reputations.
3. **Trust and audit plane:** Hyperledger Fabric issues identities, evaluates ABAC policy, preserves state needed by the next round, and records tamper-evident audit evidence. Large model vectors and raw data remain off chain.

## 4. Identity, ABAC, and on-chain state

### 4.1 Registration

Fabric CA registers each device and issues an X.509 certificate with static attributes:

`client_id`, `device_type`, `org`, `location_class`, `certificate_expiry`.

Mutable security fields are stored in Fabric world state, rather than being placed in a static certificate:

`status ∈ {ACTIVE, PROBATION, REVOKED}`, reputation `R_k`, recent-round counters, and any cohort-specific admission threshold.

This separates CA-issued identity claims from evidence-driven runtime trust.

### 4.2 Chaincode functions

| Function | Inputs | Decision / persisted record |
|---|---|---|
| `RegisterClient` | certificate identity and static attributes | creates the client state |
| `AuthorizeTraining` | client identity, round, requested action | verifies certificate, static ABAC attributes, status, and `R_k ≥ R_min,c(k)` |
| `AuthorizeUpdate` | signature, round, clipped sample count, hash of `Δ_k` | ensures one current-round authorised update and records its commitment |
| `CommitRound` | signed coordinator summary | atomically writes model hash, per-client results, cohort metadata, and next-round states |
| `QueryClientState` / `QueryAudit` | authorised reader request | returns current status and audit evidence |

`AuthorizeTraining` creates the cohort key directly from certified attributes, for example `c(k) = H(device_type || org)`. It must use the certificate values rather than a client-supplied cohort label.

## 5. One federated-learning round

1. **Eligibility gate (on chain).** The coordinator asks `AuthorizeTraining`. The policy permits a device only when its certificate is valid, required attributes satisfy policy, `status` is `ACTIVE` or `PROBATION`, and its reputation meets the applicable cohort threshold.
2. **Broadcast and local training (off chain).** Each admitted client receives `W_t`, trains locally, clips its update norm, and calculates `Δ_k = θ_k − W_t`.
3. **Update authorisation (on chain).** The client signs `{round, H(Δ_k), n_k}`. Chaincode verifies identity, round freshness, allowed status, and bounded `n_k`, then records `H(Δ_k)`. The full update is sent only to the coordinator.
4. **Cohort formation (off chain).** The coordinator groups authorised updates by their certificate-derived cohort. A cohort with fewer than `m_min` clients backs off to its parent cohort (for example, `device_type`) and then to a global reference.
5. **Robust references and scoring (off chain).** Compute a coordinate-wise median reference `r_c` per usable cohort and a global median `r_g`. For each client, calculate update features against its reference:

   `f1 = ||Δ_k||₂`, `f2 = cosine(Δ_k, r_c)`, and `f3 = ||Δ_k − r_c||₂`.

   Estimate robust cohort baselines using the median and MAD, shrunk toward global baselines when cohort support is low. Transform the features into a calibrated anomaly score `a_k ∈ [0,1]`.
6. **Cohort-capture guard (off chain).** Compare cohort reference and dispersion with the global equivalents. Produce a cohort flag `φ_c ∈ [0,1]`; a strongly divergent or internally suspicious cohort receives a higher flag. Calculate a global-reference anomaly score `a_k^glob` and set:

   `ã_k = max(a_k, φ_c × a_k^glob)`.

   This prevents a coordinated attacker-majority from appearing normal solely because it defines its own cohort baseline.
7. **Risk, aggregation, and reputation (off chain).** Calculate

   `risk_k = α × ã_k + (1 − α) × (1 − R_k)`.

   Use `risk_k` to clip, reject, or down-weight an update; apply an explicit per-cohort weight cap so a populous cohort cannot dominate aggregation. Update reputation using `ã_k`, not `risk_k`, because the latter includes historical reputation and would reinforce a past decision without new evidence.
8. **Round commitment (on chain).** The coordinator submits one compact `CommitRound` transaction. Fabric writes the model hash, cohort IDs and sizes, `(ã_k, risk_k, aggregation_weight_k, R_k, status_k)` for each participant, each `φ_c`, and a Merkle root over denied/rejected events. The next round’s eligibility gate reads this state.

## 6. Aggregation policy

The architecture uses a modular robust aggregation interface. Every evaluated rule receives only ABAC-authorised, metadata-valid updates after norm clipping.

- **Baseline:** FedAvg, retained only as a comparison point.
- **Robust comparators:** coordinate-wise median, trimmed mean, Multi-Krum, and FLTrust, as evaluated in the Byzantine-resilient edge IDS paper.
- **Proposed production path:** anomaly/reputation-weighted aggregation, with a per-cohort cap. Optionally use FLTrust as an additional trusted-direction gate when the server has a clean, representative root dataset and its maintenance is realistically justified.

Do not claim that one rule is universally best. The second paper found that robust methods can impose a clean-data cost and that the effective breakdown point depends on the attack and non-IID distribution. Select the final rule from measured results across benign and adversarial settings.

## 7. Reputation state machine

```text
                 repeated high ã_k / confirmed policy violation
ACTIVE ───────────────────────────────────────────────────────► REVOKED
  │                                                                  │
  │ elevated anomaly for sliding window                              │ administrator re-enrolment
  ▼                                                                  ▼
PROBATION ───────────── sustained low ã_k / recovery ─────────► ACTIVE
```

- `ACTIVE`: normal eligibility and the standard cohort-specific reputation threshold.
- `PROBATION`: still eligible only under stricter thresholds, lower maximum aggregation weight, and closer monitoring.
- `REVOKED`: denied at the first ABAC gate; a fresh administrative review/re-enrolment process is required.

Use asymmetric reputation updates: reduce reputation faster for repeated high-confidence malicious evidence than it rises after normal rounds. Add a sliding-window rule and minimum-evidence requirement so one noisy non-IID update does not revoke an honest device.

## 8. Evaluation design

Evaluate at least these conditions with matched random seeds and confidence intervals:

| Dimension | Required comparisons |
|---|---|
| Admission security | no ABAC vs ABAC; stale/invalid/unapproved clients denied |
| Poisoning | sign-flip, label-flip, model replacement; sweep malicious-client fraction |
| Detection design | global-only scoring vs cohort-only scoring vs cohort scoring plus capture guard and reputation |
| Non-IID severity | multiple `rho` values and partition strengths |
| Aggregation | FedAvg, median, trimmed mean, Multi-Krum, FLTrust where feasible, proposed weighted rule |
| Cohort robustness | benign cohorts, undersized-cohort fallback, and coordinated cohort capture |
| Fabric cost | policy latency, commit latency, transactions/round, ledger growth |

For every condition, report the three outcome families independently: IDS quality, malicious-client detection quality, and blockchain/access-control overhead. Evaluate false probation and false revocation explicitly, since a detector that preserves model F1 by excluding honest clients is not acceptable.

## 9. Implementation boundary

- **Phase 1 (Colab):** reproduce the CICIoT2023 preprocessing, synthetic client/cohort generation, FL simulator, attacks, detection/aggregation/reputation logic, and experiment logging. Fabric can be represented by a stable interface or mock ledger while algorithms are validated.
- **Phase 2 (Windows + Docker Desktop + WSL2):** deploy the Hyperledger Fabric test network, CA identities, Go chaincode, and a coordinator adapter. Replace the mock with actual `AuthorizeTraining`, `AuthorizeUpdate`, and single-transaction `CommitRound` calls.
- **Off-chain artefacts:** encrypted model checkpoints and update vectors; retain only hashes, outcomes, and aggregate metadata on the ledger. This avoids impractical ledger growth and keeps raw data and update tensors out of Fabric.

## 10. Novel contribution statement

The combined system is not merely “blockchain plus robust FL.” Its contribution is a layered trust mechanism: certified ABAC attributes decide who may attempt to participate; cohort-conditioned evidence judges the behaviour of admitted clients under non-IID data; capture-aware global comparison resists attacker-controlled local baselines; and Fabric makes the resulting trust lifecycle auditable and enforceable across later rounds.
