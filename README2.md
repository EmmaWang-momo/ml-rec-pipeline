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

## 如何用本地 demo 快速跑通端到端

1. 创建虚拟环境并安装依赖
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. 准备小样例数据（放 `data/sample.csv`），建议包含：`impression_id,user_id,cat1..cat10,num1..num13,click`

3. 训练单模型 DCN
```bash
python models/dcn/train.py --data_path data/sample.csv --model_dir artifacts/saved_models/dcn --nrows 20000
```

4. 转 ONNX
```bash
python models/export/convert_savedmodel_to_onnx.py --saved_model artifacts/saved_models/dcn --output artifacts/onnx_models/dcn.onnx
```

5. 启动 Router (调用 ONNXRuntime)
```bash
python realtime/router/app.py --model_onnx artifacts/onnx_models/dcn.onnx --feature_spec artifacts/saved_models/dcn/feature_spec.json
# 或者使用 uvicorn 直接运行
uvicorn realtime.router.app:app --host 0.0.0.0 --port 8080
```

6. 本地重放或压测
```bash
python stream/generator/replay.py --csv data/sample.csv --target http://localhost:8080/predict --qps 50 --nrows 1000
# 或者用 locust
locust -f stream/locust/locustfile.py --host http://localhost:8080
```

---