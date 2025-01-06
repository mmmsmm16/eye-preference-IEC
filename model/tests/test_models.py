import requests
import numpy as np
import json
import time
import pandas as pd
from typing import Dict, Any

def create_test_data(method: str = "method1", duration: float = 3.0) -> Dict[str, Any]:
    """テスト用の視線データを生成"""
    # 60Hzで3秒分のデータを生成
    num_samples = int(60 * duration)
    timestamp = np.arange(num_samples) * (1_000_000 / 60)  # マイクロ秒単位

    gaze_data = {
        "timestamp": timestamp.tolist(),
        "left_x": np.random.uniform(0, 1, num_samples).tolist(),
        "left_y": np.random.uniform(0, 1, num_samples).tolist(),
        "right_x": np.random.uniform(0, 1, num_samples).tolist(),
        "right_y": np.random.uniform(0, 1, num_samples).tolist(),
    }

    print(f"\nGenerated test data for {method}:")
    print(f"Number of samples: {num_samples}")
    print(f"Duration: {duration} seconds")
    
    return gaze_data

def test_model(model_name: str, method: str):
    """モデルのテスト"""
    print(f"\nTesting {model_name} ({method})...")
    
    # テストデータの生成
    data = create_test_data(method=method)
    
    # APIリクエスト
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "gaze_data": data,
                "model_name": model_name,
                "method": method
            },
            timeout=5
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction successful!")
            print(f"Predictions per class: {result['predictions']}")
            print(f"Predicted class: {np.argmax(result['predictions'])}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['processing_time']*1000:.2f}ms")
            print(f"Total request time: {(end_time - start_time)*1000:.2f}ms")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")

def check_server_status():
    """サーバーの状態をチェック"""
    try:
        response = requests.get("http://localhost:8000/health")
        print("\nServer health check:")
        print(json.dumps(response.json(), indent=2))
        
        response = requests.get("http://localhost:8000/models")
        print("\nAvailable models:")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Error checking server status: {str(e)}")

if __name__ == "__main__":
    print("Testing Eye Preference Prediction API...")
    
    # サーバーの状態チェック
    check_server_status()
    
    # 各モデルのテスト
    print("\nStarting model tests...")
    test_model("lstm_method1", "method1")
    test_model("transformer_method2", "method2")
