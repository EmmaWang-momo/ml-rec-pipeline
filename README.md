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
  ├─ stacked_dcn/          # 并联/堆叠 DCN variants (ensembling logic)
  ├─ export/               # SavedModel + ONNX export helpers + input spec
  └─ viz/                  # 权重/embedding 可视化 notebooks (umap/t-SNE)
realtime/                 # 在线 serving + routing + A/B 分流 + metrics client
  ├─ inference/            # Dockerized TF Serving / ONNX Runtime / Triton config examples
  ├─ router/               # feature-assembly, model client, bidding logic, REST/gRPC server
  ├─ ab_router/            # 流量分流实现（hash-based / header-based）
  └─ deploy/               # ECS/EKS / SageMaker endpoint deployment descriptors
stream/                   # Request generator + load-testing
  ├─ generator/            # Kafka (MSK) & Kinesis producers reading Avazu/iPinYou/Criteo
  ├─ locust/               # locustfile + scenarios for stress testing router/endpoints
  └─ utils/                # replay tools, rate controllers
monitoring/               # Model Monitor & infra for logging/metrics
  ├─ model_monitor/        # SageMaker Model Monitor baseline job templates & detectors
  ├─ cloudwatch/           # CloudWatch Dashboards & Alarm CloudFormation templates
  └─ observability/        # Prometheus/Grafana manifests, Opensearch ingest pipelines
report/                   # 离线/在线指标与实验报告 (notebooks + CSVs)
  ├─ offline_eval/         # HR@N, NDCG@N, eCPM simulations, counterfactual sim
  ├─ latency_throughput/   # 延迟/吞吐基准数据及绘图 notebooks
  └─ experiments/          # A/B 实验分析 + stability checks

artifacts/
  ├─ saved_models/
  └─ onnx_models/

```

---

## README (概览)

该工程用于在 AWS 上搭建一个接近 RTB（实时竞价）场景的 DCN 推荐系统，从离线训练到低延迟在线推理，再到竞价函数、实验分流与监控。仓库已按功能模块拆分，目标交付以下内容：

- 架构图（Mermaid & PNG）
- SageMaker Pipeline（`pipelines/`）: 自动化训练、评估、模型注册
- 模型实现（`models/`）: DCN 基线 + 并联/堆叠变体 + 导出工具 + 权重/embedding 可视化
- 在线服务（`realtime/`）: Dockerized 推理镜像（TF/ONNX/Triton）、Router（feature assembly + bidding + A/B）
- 请求流模拟（`stream/`）: 使用 Kafka/MSK 或 Kinesis 的 replay/generator + Locust 压测脚本
- 监控（`monitoring/`）: SageMaker Model Monitor 配置、CloudWatch Dashboard 与告警模板
- 报告（`report/`）: 离线 HR@N/NDCG@N、以及在线延迟 & 吞吐对比

---

## Architecture Diagram

下面提供一个 Mermaid 源文件（`architecture_diagram.mmd`），可用 mermaid CLI 或在线渲染器导出为 PNG/SVG。

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

## Pipelines/ (SageMaker Pipeline) — 概要

`pipelines/pipeline.py` 将实现完整的 CI 流程：

1. 数据校验（Athena/Glue 查询）
2. 特征工程（Spark on EMR 或 SageMaker Processing）
3. 训练（SageMaker Training Job，支持 Spot）
4. 模型评估（AUC / offline eCPM sim）
5. 模型注册（SageMaker Model Registry）
6. 自动部署（可选：到 SageMaker Real-Time Endpoint）

交付将包括：Pipeline 代码、IAM role 模板、以及示例参数化配置（dev/staging/prod）。

---

## Models/

- `models/dcn/`: TensorFlow DCN 的实现（cross layers + deep tower），训练脚本、超参配置、以及 local TF2 SavedModel 导出脚本（已包含 in-canvas 初版）。
- `models/stacked_dcn/`: 并联（并行）以及堆叠（串接）DCN 模型的实现和离线融合示例（平均/加权/学习器融合）。
- `models/export/`: SavedModel -> ONNX 转换脚本（使用 `tf2onnx`）与检查输入/输出名的工具。
- `models/viz/`: embedding 权重可视化（UMAP / t-SNE notebooks），以及用于生成权重分布/梯度直方图的脚本。

---

## Realtime/

- `realtime/inference/`:
  - Dockerfile 示例：TF-Serving、ONNXRuntime、Triton（含 model config）
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

- `stream/generator/`: 支持 Avazu/iPinYou/Criteo CSV replay，事件带 timestamp，可配置 QPS、burst 模式、随机 seed。
  - 支持输出到: Kinesis Data Stream (boto3), Kafka (confluent-kafka-python)
- `stream/locust/`: Locust 场景文件，用于施加 HTTP/gRPC 负载到 Router/Inference endpoints，并可读取 metrics 输出为 CSV

---

## Monitoring/

- `monitoring/model_monitor/`: SageMaker Model Monitor baseline job (baseline windows), 以及 drift detectors（pCTR vs observed CTR）
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
