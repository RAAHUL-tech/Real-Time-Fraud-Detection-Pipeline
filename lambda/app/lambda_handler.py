import json
import onnxruntime as ort
import numpy as np
import os
import time

# Load ONNX model once at cold start
model_path = os.path.join(os.path.dirname(__file__), "fraud_model.onnx")
session = ort.InferenceSession(model_path) 
input_name = session.get_inputs()[0].name


def lambda_handler(event, context):
    for record in event["Records"]:
        try:
            data = json.loads(record["body"])
            print(data)
            # Extract and validate features
            features = data.get("features")
            if not features or len(features) != 29:
                raise ValueError("Invalid input features")

            # Run ONNX inference
            inputs = np.array(features, dtype=np.float32).reshape(1, -1)
            start_time = time.time()
            output = session.run(None, {input_name: inputs})
            inference_time = (time.time() - start_time) * 1000  # ms

            prediction = float(output[0][0][0])
            print(f"[INFO] Prediction: {prediction:.4f}, Inference time: {inference_time:.2f} ms")

        except Exception as e:
            print(f"[ERROR] Failed processing record: {e}")

    return {
        "statusCode": 200,
        "body": json.dumps("Inference completed successfully")
    }
