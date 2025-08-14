# file: tf_server.py
import tensorflow as tf
import numpy as np
import math
from fastapi import FastAPI
from pydantic import BaseModel

# 加载 SavedModel
model = tf.keras.models.load_model("/Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/saved_models/dcn")

# FastAPI 实例
app = FastAPI()


# 请求格式
class PredictRequest(BaseModel):
    features: list  # 二维数组，比如 [[0.1, 0.2, ...]]


@app.post("/predict")
def predict(req: PredictRequest):
    # 转为 numpy
    inputs = np.array([req.features,], dtype=np.float32)
    # 模型推理
    preds = model(inputs)
    # 转为 python list
    try:
        print('origin result: ', preds.numpy())
        result = float(preds.numpy().tolist()[0][0])
    except Exception as err:
        result = -1
    if math.isnan(result):
        result = -2
    print('return result: ', result)
    return {"predictions": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)