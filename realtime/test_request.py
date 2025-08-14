import requests
import numpy as np

# # 构造测试输入
sample_data = [0 for i in range(122)]

# 请求
resp = requests.post(
    "http://localhost:8501/predict",
    json={"features": sample_data}
)

print("Response:", resp.json())