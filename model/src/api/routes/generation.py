from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..schemas.generation import GenerationRequest, GenerationResponse, GeneratedImage
from ...services.stable_diffusion_service import StableDiffusionService
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# 保存先のベースディレクトリ
BASE_DIR = "data/experiment_sessions"
EXPLICIT_DIR = os.path.join(BASE_DIR, "manual_evaluation")
GAZE_DIR = os.path.join(BASE_DIR, "gaze_evaluation")

sd_service = StableDiffusionService()

def get_session_dir(session_id: str, interaction_mode: str) -> str:
    """セッションの保存先ディレクトリを取得"""
    base = GAZE_DIR if interaction_mode == 'gaze' else EXPLICIT_DIR
    return os.path.join(base, session_id)

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")

@router.post("/generate")
async def generate_images(request: GenerationRequest):
    try:
        logger.debug(f"Received generation request: {request}")

        # インタラクションモードに基づいてセッションディレクトリを決定
        session_dir = get_session_dir(request.session_id, request.interaction_mode)
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
            generation=request.generation,
            session_id=request.session_id
        )

        # 生成された画像を保存して応答を準備
        image_data = []
        for idx, (image, latent) in enumerate(generated_results):
            filename = f"image_{idx}.png"
            filepath = os.path.join(step_dir, filename)
            
            try:
                sd_service.save_image(image, filepath)
                logger.debug(f"Saved image to {filepath}")
            except Exception as e:
                logger.error(f"Error saving image {idx}: {str(e)}")
                raise

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

@router.get("/session-data/{session_id}/step_{step}/{filename}")
async def get_image(session_id: str, step: str, filename: str):
    try:
        # まずgazeディレクトリを確認
        filepath = os.path.join(GAZE_DIR, session_id, step, filename)
        if not os.path.exists(filepath):
            # 見つからない場合はexplicitディレクトリを確認
            filepath = os.path.join(EXPLICIT_DIR, session_id, step, filename)
        
        logger.debug(f"Attempting to serve image from: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"Image not found: {filepath}")
            raise HTTPException(status_code=404, detail=f"Image not found: {filepath}")
            
        return FileResponse(filepath, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
