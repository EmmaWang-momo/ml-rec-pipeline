# RTB-style DCN Recommendation System (AWS)

**Deliverables (repo root)**

```
README.md
architecture_diagram.mmd
pipelines/                # SageMaker Pipeline implementation (training, validation, model catalog)
  ├─ pipeline.py
  ├─ infra/                # Terraform/CloudFormation snippets for SageMaker roles, S3, ECR
  └─ templates/
models/                   # DCN model code, stacking/parallel variants, export & viz
  ├─ dcn/                  # TF DCN implementation (cross + deep) + training scripts
  ├─ stacked_dcn/          # Parallel/stacked DCN variants (ensembling logic)
  ├─ export/               # SavedModel + ONNX export helpers + input spec
  └─ viz/                  # Weight/embedding visualization notebooks (UMAP/t-SNE)
realtime/                 # Online serving + routing + A/B splitting + metrics client
  ├─ inference/            # Dockerized TF Serving / ONNX Runtime / Triton config examples
  ├─ router/               # feature assembly, model client, bidding logic, REST/gRPC server
  ├─ ab_router/            # Traffic splitting implementations (hash-based / header-based)
  └─ deploy/               # ECS/EKS / SageMaker endpoint deployment descriptors
stream/                   # Request generator + load-testing
  ├─ generator/            # Kafka (MSK) & Kinesis producers reading Avazu/iPinYou/Criteo
  ├─ locust/               # locustfile + scenarios for stress testing router/endpoints
  └─ utils/                # replay tools, rate controllers
monitoring/               # Model Monitor & infra for logging/metrics
  ├─ model_monitor/        # SageMaker Model Monitor baseline job templates & detectors
  ├─ cloudwatch/           # CloudWatch Dashboards & Alarm CloudFormation templates
  └─ observability/        # Prometheus/Grafana manifests, Opensearch ingest pipelines
report/                   # Offline/online metrics and experiment reports (notebooks + CSVs)
  ├─ offline_eval/         # HR@N, NDCG@N, eCPM simulations, counterfactual sim
  ├─ latency_throughput/   # Latency/throughput benchmark data and plotting notebooks
  └─ experiments/          # A/B experiment analysis + stability checks

artifacts/
  ├─ saved_models/
  └─ onnx_models/

```

---

## README (Overview)

This project builds an RTB-like DCN recommendation system on AWS, covering the full pipeline from offline training to low-latency online inference, bidding logic, experiment traffic splitting, and monitoring. The repository is modularized by functionality and delivers:

Architecture diagram (Mermaid & PNG)
SageMaker Pipeline (pipelines/): automated training, evaluation, and model registration
Model implementations (models/): DCN baseline + parallel/stacked variants + export tools + weight/embedding visualization
Online serving (realtime/): Dockerized inference images (TF/ONNX/Triton), Router (feature assembly + bidding + A/B)
Request streaming simulation (stream/): Kafka/MSK or Kinesis replay/generator + Locust load testing scripts
Monitoring (monitoring/): SageMaker Model Monitor configs, CloudWatch dashboards and alarm templates
Reports (report/): offline HR@N/NDCG@N, and online latency & throughput comparisons

---

## Architecture Diagram

Mermaid source file (architecture_diagram.mmd), renderable to PNG/SVG:

```mmd
flowchart LR
  subgraph DATA_LAKE
    S3[(S3)]
    Glue[Glue/EMR]
  end
  subgraph TRAINING
    SageMaker["SageMaker Training
Pipeline"]
    FeatureStore["Feature Store"]
  end
  subgraph STREAM
    Kinesis[Kinesis/MSK]
    Generator[Request Generator]
  end
  subgraph SERVING
    Router[Router Service]
-- feature assemble --> Redis[Redis Cache]
    ModelServing[TF/ONNX/Triton]
    AB[AB Router]
  end
  subgraph MONITOR
    CloudWatch[CloudWatch/Grafana]
    ModelMonitor[SageMaker Model Monitor]
  end

  S3 --> SageMaker
  Glue --> S3
  SageMaker --> FeatureStore
  Generator --> Kinesis --> Router --> ModelServing
  Router --> Redis
  ModelServing --> CloudWatch
  FeatureStore --> Router
  ModelMonitor --> CloudWatch
  CloudWatch --> Alert[Alerting/Pager]

```

---

## Pipelines/ (SageMaker Pipeline) — Overview

`pipelines/pipeline.py` implements the complete CI/CD flow:

Data validation (Athena/Glue queries)
Feature engineering (Spark on EMR or SageMaker Processing)
Training (SageMaker Training Job, Spot support)
Model evaluation (AUC / offline eCPM simulation)
Model registration (SageMaker Model Registry)
Optional automatic deployment (SageMaker Real-Time Endpoint)

Deliverables include: pipeline code, IAM role templates, and example parameterized configs (dev/staging/prod).

---

## Models/

- `models/dcn/`: TensorFlow DCN implementation (cross layers + deep tower), training scripts, hyperparameter configs, and local TF2 SavedModel export scripts.
- `models/stacked_dcn/`: Parallel and stacked DCN implementations and offline fusion examples (average/weighted/learner-based).
- `models/export/`:  SavedModel → ONNX conversion scripts (tf2onnx) and input/output name checking tools.
- `models/viz/`: Embedding weight visualization (UMAP / t-SNE notebooks) and scripts for generating weight distribution/gradient histograms.

---

## Realtime/

- `realtime/inference/`:
  - Example Dockerfiles: TF-Serving, ONNXRuntime, Triton (with model config)
  - Perf tips：serve with gRPC, use batcher/queue, keep instances warm
- `realtime/router/`:
  - FastAPI / gRPC router that:
    - assembles features (reads Redis/Dynamo/FeatureStore),
    - calls model endpoint (gRPC/HTTP),
    - runs bidding function and returns bid response,
    - logs trace + metrics to CloudWatch/Prometheus.
  - Includes hash-based A/B routing middleware and AB traffic controller.
- `realtime/deploy/`: ecs_task_def.json, eks helm values, sagemaker endpoint config

Metrics collected: pCTR distribution, latency (P50/P90/P99), win-rate, eCPM, errors.

---

## Stream/

stream/generator/: Supports Avazu/iPinYou/Criteo CSV replay with timestamps, configurable QPS, burst mode, and random seeds. Output to: Kinesis Data Stream (boto3), Kafka (confluent-kafka-python)
stream/locust/: Locust scenario files for applying HTTP/gRPC load to Router/Inference endpoints and exporting metrics to CSV

---

## Monitoring/

- `monitoring/model_monitor/`: SageMaker Model Monitor baseline job (baseline windows), and drift detectors（pCTR vs observed CTR）
- `monitoring/cloudwatch/`: CloudFormation templates for dashboards & alarms (latency, error rate, spend pacing)
- `monitoring/observability/`: Prometheus exporter for router & model server, Grafana dashboard JSON

---

## Report/

- `report/offline_eval/`: Notebook and scripts to compute HR@N, NDCG@N, and offline counterfactual auction simulations (using historical `market_price` column to compute win-rate and realized eCPM)
- `report/latency_throughput/`: Jupyter notebook with plots comparing SageMaker Endpoint vs Triton (ECS/EKS) vs ONNXRuntime (EC2) under load (locust outputs)

---

## Quickstart (Dev machine)

1. Clone repo
2. Create Python env: `python -m venv venv && source venv/bin/activate` 
3. `pip install -r requirements-dev.txt` (requirements will include: tensorflow, tf2onnx, boto3, fastapi, uvicorn, confluent-kafka, locust)
4. Put a sample CSV into `data/` (or provide S3 path via env)
5. Run a quick local train (small nrows):
   ```bash
   python models/dcn/train.py --data_path data/sample.csv --nrows 10000 --model_dir ./artifacts/dev_model
   ```
6. Export to SavedModel & ONNX:
   ```bash
   python models/export/convert_savedmodel_to_onnx.py --saved_model ./artifacts/dev_model/saved_model --output ./artifacts/dev_model/model.onnx
   ```
7. Start local router pointing to local ONNX runtime service (docker):
   ```bash
   # start onnxruntime server (example image)
   docker run --rm -p 8001:8001 my-onnx-server:latest
   # start router
   uvicorn realtime/router/app:app --host 0.0.0.0 --port 8080
   ```
8. Start request generator to stream to router (Kinesis or direct HTTP):
   ```bash
   python stream/generator/replay.py --csv data/sample.csv --target http://localhost:8080/predict --qps 50
   ```

---

## Next steps / What I can deliver next (pick one or more):

1. Scaffold the actual file tree and populate `pipelines/pipeline.py` (SageMaker Pipeline) — includes example PipelineDefinition and local runner.
2. Produce `models/` TF-DCN full training notebook + hyperparam config (already partially in canvas). Add stacked/parallel variant.
3. Implement `realtime/router` (FastAPI) + a Dockerfile and a simple ONNXRuntime server example.
4. Implement `stream/generator` with Kinesis + Locust scenarios.
5. Add Monitoring CloudFormation / Terraform templates + a Grafana dashboard JSON.
6. Run a small end-to-end demo locally and produce `report/` with sample HR@10 and latency plots.
