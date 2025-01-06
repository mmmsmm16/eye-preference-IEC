from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..schemas.generation import GenerationRequest, GenerationResponse, GeneratedImage
from ...services.stable_diffusion_service import StableDiffusionService
import os
import shutil
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# 保存先のベースディレクトリを更新
BASE_DIR = "data/experiment_sessions"

sd_service = StableDiffusionService()

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")

@router.post("/generate")
async def generate_images(request: GenerationRequest):
    """画像生成エンドポイント"""
    try:
        logger.debug(f"Received generation request: {request}")

        # セッションとステップのディレクトリを作成
        session_dir = os.path.join(BASE_DIR, request.session_id)
        step_dir = os.path.join(session_dir, f"step_{request.generation}")
        ensure_directory_exists(step_dir)
        logger.debug(f"Using directory: {step_dir}")

        # base_vectorがある場合は変換
        base_latent = None
        if request.base_vector:
            base_latent = sd_service.base64_to_latent(request.base_vector)
            logger.debug("Successfully converted base vector")

        # 画像生成
        generated_results = await sd_service.generate_images(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            base_latent=base_latent,
            num_images=request.num_images,
            generation=request.generation
        )

        # 生成された画像を保存
        image_data = []
        for idx, (image, latent) in enumerate(generated_results):
            # タイムスタンプベースのファイル名を削除し、シンプルな連番に変更
            filename = f"image_{idx}.png"
            filepath = os.path.join(step_dir, filename)
            sd_service.save_image(image, filepath)
            logger.debug(f"Saved image to {filepath}")

            # レスポンスデータの作成
            latent_base64 = sd_service.latent_to_base64(latent)
            image_data.append(GeneratedImage(
                url=f"/session-data/{request.session_id}/step_{request.generation}/{filename}",
                latent_vector=latent_base64,
                prompt=request.prompt,
                generation=request.generation
            ))

        response = GenerationResponse(
            images=image_data,
            generation=request.generation,
            base_prompt=request.prompt
        )
        logger.debug("Successfully generated response")
        return response

    except Exception as e:
        logger.error(f"Error generating images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session-data/{session_id}/{step}/{filename}")
async def get_image(session_id: str, step: str, filename: str):
    """生成された画像を取得するエンドポイント"""
    try:
        filepath = os.path.join(BASE_DIR, session_id, step, filename)
        logger.debug(f"Attempting to serve image: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"Image not found: {filepath}")
            raise HTTPException(status_code=404, detail=f"Image not found: {filepath}")
            
        return FileResponse(filepath)
        
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
