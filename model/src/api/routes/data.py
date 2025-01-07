from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import json
import logging
import csv
import shutil
import glob
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["data"])

# 保存先のベースディレクトリ
BASE_DIR = "data/experiment_sessions"
MANUAL_DIR = os.path.join(BASE_DIR, "manual_evaluation")
GAZE_DIR = os.path.join(BASE_DIR, "gaze_evaluation")

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
    selection: Dict[str, Any]
    evaluationMethod: str  # 追加: 評価モードを含める

class SessionMetadata(BaseModel):
    evaluationMethod: str = Field(..., description="Evaluation method (gaze/explicit)")
    modelId: Optional[str] = None
    prompt: str = Field(..., description="Generation prompt")
    negativePrompt: Optional[str] = None
    fixedPlacement: bool = Field(..., description="Whether image placement is fixed")
    eyeTrackerConnected: bool = Field(..., description="Eye tracker connection status")
    timestamp: str = Field(..., description="Current timestamp in ISO format")
    startTime: str = Field(..., description="Session start time in Japan time")
    endTime: Optional[str] = Field(None, description="Session end time in Japan time")
    sessionDurationSeconds: Optional[float] = Field(None, description="Session duration in seconds")
    finalSelectedImage: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "evaluationMethod": "gaze",
                "modelId": "lstm_method1",
                "prompt": "a serene landscape",
                "negativePrompt": None,
                "fixedPlacement": True,
                "eyeTrackerConnected": True,
                "timestamp": datetime.now().isoformat(),
                "startTime": "2024-01-07 15:30:45",  # サンプルの日本時間
                "endTime": "2024-01-07 15:35:20",    # サンプルの日本時間
                "sessionDurationSeconds": 275.0,      # 秒単位のduration
                "finalSelectedImage": None
            }
        }

class SessionMetadataRequest(BaseModel):
    sessionId: str = Field(..., description="Session identifier")
    metadata: SessionMetadata


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

def get_session_dir(evaluation_method: str, session_id: str) -> str:
    """評価方法に基づいて適切なセッションディレクトリを返す"""
    # evaluation_methodが'explicit'の場合は'manual'として扱う
    if evaluation_method == 'explicit':
        evaluation_method = 'manual'
    
    # ベースディレクトリを選択
    base_dir = GAZE_DIR if evaluation_method == 'gaze' else MANUAL_DIR
    
    # セッションディレクトリのパスを生成
    session_dir = os.path.join(base_dir, f"{session_id}")
    
    return session_dir

def ensure_directory_exists(directory: str):
    """ディレクトリが存在することを確認し、必要に応じて作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def save_final_image(image_url: str, target_path: str) -> None:
    """最終選択画像を保存"""
    try:
        # 元の画像をコピー
        source_path = os.path.join(BASE_DIR, image_url.split('/session-data/')[-1])
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            logger.debug(f"Final image saved to: {target_path}")
        else:
            logger.error(f"Source image not found: {source_path}")
            raise FileNotFoundError(f"Source image not found: {source_path}")
    except Exception as e:
        logger.error(f"Error saving final image: {str(e)}")
        raise


@router.get("/get-next-session-number/{evaluation_mode}")
async def get_next_session_number(evaluation_mode: str):
    """次のセッション番号を取得"""
    try:
        # 評価モードに基づいてディレクトリを選択
        base_dir = GAZE_DIR if evaluation_mode == 'gaze' else MANUAL_DIR
        
        # セッションディレクトリの一覧を取得
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            return {"nextNumber": 1}
            
        existing_sessions = [
            d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        
        # 既存の番号を抽出
        numbers = []
        for session in existing_sessions:
            try:
                num = int(session)
                numbers.append(num)
            except ValueError:
                continue
        
        # 次の番号を決定
        next_number = 1 if not numbers else max(numbers) + 1
        
        return {"nextNumber": next_number}
    
    except Exception as e:
        logger.error(f"Error getting next session number: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# エンドポイントのパスを修正
@router.post("/step-data")
async def save_step_data(request: SaveDataRequest):
    logger.debug(f"Received save request for session {request.sessionId}")
    
    try:
        # 評価方法に基づいてセッションディレクトリを決定
        session_dir = get_session_dir(request.evaluationMethod, request.sessionId)
        step_dir = os.path.join(session_dir, f"step_{request.stepId}")
        ensure_directory_exists(step_dir)

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
                "json": json_path,
                "step_dir": step_dir
            }
        }

    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/session-metadata")
async def save_session_metadata(request: SessionMetadataRequest):
    """セッションのメタデータを保存"""
    logger.debug(f"Received session metadata for session {request.sessionId}")
    
    try:
        # 評価方法に基づいて適切なディレクトリを選択
        session_dir = get_session_dir(
            request.metadata.evaluationMethod,
            request.sessionId
        )
        ensure_directory_exists(session_dir)

        # メタデータをJSONとして保存
        metadata_path = os.path.join(session_dir, "session_info.json")
        metadata_dict = request.metadata.dict()
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.debug(f"Saved session metadata to {metadata_path}")

        # 最終選択画像が含まれている場合は保存
        if request.metadata.finalSelectedImage:
            final_image_name = "final_selected_image.png"
            final_image_path = os.path.join(session_dir, final_image_name)
            image_url = request.metadata.finalSelectedImage.get('src')
            if image_url:
                save_final_image(image_url, final_image_path)

        return {
            "status": "success",
            "message": "Session metadata saved successfully",
            "path": metadata_path
        }

    except Exception as e:
        logger.error(f"Error saving session metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 起動時にディレクトリ構造を作成
@router.on_event("startup")
async def startup_event():
    """必要なディレクトリ構造を作成"""
    try:
        # ベースディレクトリを作成
        ensure_directory_exists(BASE_DIR)
        # 各評価方式のディレクトリを作成
        ensure_directory_exists(MANUAL_DIR)
        ensure_directory_exists(GAZE_DIR)
        logger.info(f"Directory structure initialized: {BASE_DIR}")
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        raise
