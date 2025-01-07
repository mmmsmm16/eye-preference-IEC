from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, validator

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    base_vector: Optional[str] = None
    num_images: int = Field(4, description="Number of images to generate")
    generation: int
    session_id: str
    interaction_mode: str  # 追加: 'gaze' または 'explicit'
    
    @validator('prompt')
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('prompt cannot be empty')
        return v.strip()
    
    @validator('num_images')
    def validate_num_images(cls, v):
        if v < 1 or v > 4:
            raise ValueError('num_images must be between 1 and 4')
        return v
    
    @validator('generation')
    def validate_generation(cls, v):
        if v < 0:
            raise ValueError('generation must be non-negative')
        return v
    
    @validator('interaction_mode')
    def validate_interaction_mode(cls, v):
        if v not in ['gaze', 'explicit']:
            raise ValueError('interaction_mode must be either "gaze" or "explicit"')
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "a serene landscape with mountains",
            "negative_prompt": None,
            "base_vector": None,
            "num_images": 4,
            "generation": 0,
            "interaction_mode": "explicit"
        }
    })

class GeneratedImage(BaseModel):
    url: str = Field(..., description="URL or path to the generated image")
    latent_vector: str = Field(..., description="Base64 encoded latent vector")
    prompt: str = Field(..., description="Prompt used to generate this image")
    generation: int = Field(..., description="Generation number of this image")

class GenerationResponse(BaseModel):
    images: List[GeneratedImage]
    generation: int
    base_prompt: str
