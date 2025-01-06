from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import json
import logging
import csv

logger = logging.getLogger(__name__)
router = APIRouter(tags=["data"])

# 共通のベースディレクトリを使用
BASE_DIR = "data/experiment_sessions"  # すべてのデータを保存する共通ディレクトリ

class SelectionInfo(BaseModel):
    selectedImageIndex: Optional[int] = None
    predictedIndex: Optional[int] = None
    predictionConfidence: Optional[float] = None

class SaveDataRequest(BaseModel):
    sessionId: str
    stepId: int
    timestamp: str
    gazeData: Optional[Dict[str, Any]] = None
    images: List[Dict[str, Any]]
    prompt: str
    generation: int
    selection: SelectionInfo  # 選択情報を追加

def save_gaze_data_as_csv(gaze_data: List[Dict], file_path: str):
    """視線データをCSVファイルとして保存する関数"""
    try:
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'left_x', 'left_y', 'right_x', 'right_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data_point in gaze_data:
                writer.writerow({
                    'timestamp': data_point.get('timestamp', ''),
                    'left_x': data_point.get('left_x', ''),
                    'left_y': data_point.get('left_y', ''),
                    'right_x': data_point.get('right_x', ''),
                    'right_y': data_point.get('right_y', '')
                })
        logger.debug(f"Successfully saved gaze data to CSV: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving gaze data to CSV: {str(e)}")
        raise

# エンドポイントのパスを修正
@router.post("/step-data", response_model=Dict[str, Any])
async def save_step_data(request: SaveDataRequest):
    logger.debug(f"Received save request for session {request.sessionId}")
    
    try:
        # セッションとステップのディレクトリパスを構築
        session_dir = os.path.join(BASE_DIR, request.sessionId)
        step_dir = os.path.join(session_dir, f"step_{request.stepId}")
        os.makedirs(step_dir, exist_ok=True)

        # 基本データをJSONとして保存
        step_data = request.dict()
        if 'gazeData' in step_data and step_data['gazeData'] and 'data' in step_data['gazeData']:
            gaze_data = step_data['gazeData'].pop('data')
            # 視線データをCSVとして保存
            if gaze_data:
                csv_path = os.path.join(step_dir, "gaze_data.csv")
                with open(csv_path, 'w', newline='') as f:
                    f.write("timestamp,left_x,left_y,right_x,right_y\n")
                    for point in gaze_data:
                        f.write(f"{point['timestamp']},{point['left_x']},{point['left_y']},{point['right_x']},{point['right_y']}\n")
                logger.debug(f"Saved gaze data to CSV: {csv_path}")

        # メインのJSONデータを保存
        json_path = os.path.join(step_dir, "step_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2)
        logger.debug(f"Saved step data to JSON: {json_path}")

        return {
            "status": "success",
            "message": "Data saved successfully",
            "paths": {
                "base_dir": session_dir,
                "step_dir": step_dir,
                "json": json_path
            }
        }

    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# 起動時にディレクトリを作成
@router.on_event("startup")
async def startup_event():
    os.makedirs(BASE_DIR, exist_ok=True)
    logger.info(f"Ensuring experiment directory exists: {BASE_DIR}")

@router.get("/status")
async def check_status():
    return {
        "status": "healthy",
        "base_dir": BASE_DIR,
        "base_dir_exists": os.path.exists(BASE_DIR)
    }
