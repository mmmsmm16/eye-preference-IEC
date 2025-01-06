import requests
import numpy as np
import pandas as pd
from typing import Dict, Any
import json

def test_model_loading():
    """モデルのロードをテスト"""
    try:
        response = requests.get("http://localhost:8000/models")
        print("\nAvailable models:")
        print(json.dumps(response.json(), indent=2))
        assert response.status_code == 200
    except Exception as e:
        print(f"Error in model loading test: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        raise

def test_health():
    """ヘルスチェック"""
    try:
        response = requests.get("http://localhost:8000/health")
        print("\nHealth status:")
        print(json.dumps(response.json(), indent=2))
        assert response.status_code == 200
    except Exception as e:
        print(f"Error in health check: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        raise

def create_sample_gaze_data(method: str = "method1", duration: float = 3.0) -> Dict[str, Any]:
    """サンプルの視線データを生成"""
    # 60Hzで3秒分のデータを生成
    num_samples = int(60 * duration)
    timestamp = np.arange(num_samples) * (1_000_000 / 60)  # マイクロ秒単位

    print(f"\nGenerating {method} data:")
    print(f"Number of samples: {num_samples}")
    print(f"Duration: {duration} seconds")

    if method == "method1":
        # 4次元の視線データ (left_x, left_y, right_x, right_y)
        data = {
            "timestamp": timestamp.tolist(),
            "left_x": np.random.uniform(0, 1, num_samples).tolist(),
            "left_y": np.random.uniform(0, 1, num_samples).tolist(),
            "right_x": np.random.uniform(0, 1, num_samples).tolist(),
            "right_y": np.random.uniform(0, 1, num_samples).tolist(),
        }
    else:  # method2
        # 3次元の注視点データ (x, y, duration)
        num_fixations = 10
        data = {
            "timestamp": timestamp[:num_fixations].tolist(),
            "x": np.random.uniform(0, 1, num_fixations).tolist(),
            "y": np.random.uniform(0, 1, num_fixations).tolist(),
            "duration": np.random.uniform(0.1, 0.5, num_fixations).tolist(),
        }
    
    return data

def test_prediction(model_name: str, method: str):
    """予測のテスト"""
    print(f"\nTesting prediction for {model_name} ({method})")
    
    # サンプルデータの生成
    data = create_sample_gaze_data(method=method)
    
    # リクエストデータの作成
    request_data = {
        "gaze_data": data,
        "model_name": model_name,
        "method": method
    }
    
    print("\nSending request with data:")
    print(json.dumps(request_data, indent=2))
    
    # 予測リクエスト
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=request_data,
            timeout=5
        )
        
        print(f"\nResponse status code: {response.status_code}")
        print("Response content:")
        print(json.dumps(response.json(), indent=2))
        
        assert response.status_code == 200, f"Unexpected status code: {response.status_code}, Response: {response.text}"
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error during prediction test: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        raise

if __name__ == "__main__":
    print("Starting API tests...")
    
    # 基本的なエンドポイントのテスト
    test_health()
    test_model_loading()
    
    # 各モデルの予測テスト
    test_prediction("lstm_method1", "method1")
    test_prediction("transformer_method2", "method2")
