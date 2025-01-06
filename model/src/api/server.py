from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
import logging
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from .routes import generation
from .routes import data
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.staticfiles import StaticFiles

# プロジェクトのルートディレクトリをPYTHONPATHに追加
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 相対インポートを絶対インポートに変更
from src.models.model_loader import ModelLoader
from src.data.preprocess import (
    preprocess_method1, 
    preprocess_method2, 
    check_gaze_data_quality,
    trim_based_on_fixations
)

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 設定のデフォルト値
MODEL_CONFIG = {
    "model_dir": {
        "lstm": {
            "method1": Path("/app/src/models/lstm/method1"),  # パスを修正
            "method2": Path("/app/src/models/lstm/method2")
        },
        "transformer": {
            "method1": Path("/app/src/models/transformer/method1"),
            "method2": Path("/app/src/models/transformer/method2")
        }
    },
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 1,
    "default_method": "method1",
}

# 共通の定数
DATA_DIR = "data/experiment_sessions"

# モデルローダーのインスタンス
model_loader = None

class GazeData(BaseModel):
    timestamp: List[int]
    left_x: List[float]
    left_y: List[float]
    right_x: List[float]
    right_y: List[float]

class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence: float
    processing_time: float
    method: str
    model_type: str

def process_gaze_data(
    gaze_data: Dict[str, List],
    method: str = "method1"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    視線データの前処理を行う関数
    
    :param gaze_data: 生の視線データ
    :param method: 前処理手法 ("method1" or "method2")
    :return: 処理済みデータと統計情報
    """
    # DataFrameの作成
    df = pd.DataFrame({
        'timestamp': gaze_data['timestamp'],
        'left_x': gaze_data['left_x'],
        'left_y': gaze_data['left_y'],
        'right_x': gaze_data['right_x'],
        'right_y': gaze_data['right_y'],
    })
    
    logger.debug(f"Created DataFrame with shape: {df.shape}")
    
    # 前処理の実行
    if method == "method1":
        logger.debug("Using method1 preprocessing")
        processed_data, stats = trim_based_on_fixations(df)
        if processed_data.empty:
            raise ValueError("Preprocessing resulted in empty data")
        final_data = processed_data[['left_x', 'left_y', 'right_x', 'right_y']].values
    else:  # method2
        logger.debug("Using method2 preprocessing")
        processed_data, stats = trim_based_on_fixations(df)
        if processed_data.empty:
            raise ValueError("Preprocessing resulted in empty data")
        final_data = preprocess_method2(processed_data)
        if final_data.size == 0:
            raise ValueError("Preprocessing resulted in empty data")
    
    logger.debug(f"Preprocessing completed. Output shape: {final_data.shape}")
    return final_data, stats

app = FastAPI(title="IEC API")

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルの設定
app.mount("/session-data", StaticFiles(directory="data/experiment_sessions"), name="session_data")

# ディレクトリ作成関数
def ensure_directories():
    """必要なディレクトリを作成"""
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {DATA_DIR}")


# タイムアウト設定を追加
@app.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    response.headers["Keep-Alive"] = "timeout=300"
    return response

# ディレクトリを作成してから静的ファイルのマウントを設定
ensure_directories()
app.mount("/session-data", StaticFiles(directory=DATA_DIR), name="session_data")

# ルーターの登録
app.include_router(generation.router)
app.include_router(data.router, prefix="/save-data", tags=["data"])

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.get("/")
async def root():
    return {"message": "API is running"}


# デバッグ用: 使用可能なルートを表示
@app.on_event("startup")
async def startup_event():
    routes = [
        f"{route.path} [{', '.join(route.methods)}]"
        for route in app.routes
        if hasattr(route, "methods")  # Mountedルートを除外
    ]
    logger.debug("Available routes:")
    for route in routes:
        logger.debug(f"  {route}")

    """アプリケーション起動時にモデルをロード"""
    global model_loader
    try:
        model_dir = MODEL_CONFIG["model_dir"]["lstm"]["method1"]
        logger.info(f"Looking for models in: {model_dir}")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # モデルローダーの初期化
        model_loader = ModelLoader(
            model_dir=model_dir,
            device=MODEL_CONFIG["device"]
        )

        # 利用可能なモデルを確認
        available_models = model_loader.list_available_models()
        if not available_models:
            # モデルファイルの一覧を出力
            files = list(Path(model_dir).glob('*'))
            logger.error(f"No models found in directory. Files found: {[str(f) for f in files]}")
            raise RuntimeError("No available models found")

        logger.info(f"Available models: {available_models}")

        # 最初のモデルをロード
        model_loader.load_model(available_models[0]['name'])
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to initialize model loader: {str(e)}")
        raise
        
@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": model_loader is not None,
        "available_models": model_loader.list_available_models() if model_loader else [],
        "data_directory": {
            "path": DATA_DIR,
            "exists": os.path.exists(DATA_DIR),
            "is_writable": os.access(DATA_DIR, os.W_OK) if os.path.exists(DATA_DIR) else False
        }
    }

@app.post("/predict")
async def predict(
    gaze_data: GazeData,
    model_name: Optional[str] = None,
    method: Optional[str] = None
) -> PredictionResponse:
    """視線データから好みの画像を予測"""
    try:
        # パラメータの検証
        if not model_loader:
            raise HTTPException(status_code=500, detail="Model loader not initialized")
        
        if not method or method not in ["method1", "method2"]:
            method = "method1"  # デフォルト値を設定
        
        logger.info(f"Received prediction request with {len(gaze_data.timestamp)} samples")
        
        # データの前処理
        try:
            processed_data, stats = process_gaze_data(
                {
                    'timestamp': gaze_data.timestamp,
                    'left_x': gaze_data.left_x,
                    'left_y': gaze_data.left_y,
                    'right_x': gaze_data.right_x,
                    'right_y': gaze_data.right_y,
                },
                method=method
            )
            
            logger.info(f"Preprocessing stats: {stats}")
            logger.info(f"Processed data shape: {processed_data.shape}")
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to preprocess data: {str(e)}"
            )

        # モデルの選択と読み込み
        if model_name:
            try:
                model_loader.load_model(model_name)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load model {model_name}: {str(e)}"
                )
        elif not model_loader.current_model:
            available_models = model_loader.list_available_models()
            if not available_models:
                raise HTTPException(status_code=500, detail="No models available")
            model_loader.load_model(available_models[0]['name'])

        # 推論の実行
        try:
            # データをテンソルに変換
            tensor_data = torch.tensor(processed_data, dtype=torch.float32)
            tensor_data = tensor_data.unsqueeze(0)  # バッチ次元の追加
            
            logger.debug(f"Input tensor shape: {tensor_data.shape}")
            
            # デバイスに転送
            tensor_data = tensor_data.to(MODEL_CONFIG["device"])
            
            # 推論実行
            with torch.no_grad():
                output = model_loader.current_model(tensor_data)
                probabilities = torch.softmax(output, dim=1)

            # 結果の整形
            predictions = probabilities[0].cpu().numpy().tolist()
            confidence = float(torch.max(probabilities[0]).cpu())

            logger.info(f"Prediction complete. Confidence: {confidence}")

            return PredictionResponse(
                predictions=predictions,
                confidence=confidence,
                processing_time=0.0,  # 実際の処理時間の計測は省略
                method=method,
                model_type=model_loader.current_model_info["type"]
            )

        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to run inference: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process gaze data: {str(e)}"
        )
    
# エラーハンドリング
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler caught: {str(exc)}")
    return {
        "status": "error",
        "message": str(exc),
        "path": request.url.path
    }
if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        workers=1
    )
