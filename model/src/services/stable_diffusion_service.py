import os
from typing import List, Optional, Tuple
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import io
import base64
import numpy as np

class StableDiffusionService:
    def __init__(self):
        self.model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe = self.pipe.to("cuda")
        self.generator = torch.Generator(device="cuda")
        self.latent_shape = (1, 4, 64, 64)
        self.current_sigma = 1.0  # 初期の突然変異強度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "get_data"
        
    def generate_initial_noise(
        self,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """初期ノイズを生成"""
        if seed is not None:
            self.generator.manual_seed(seed)
        return randn_tensor(
            self.latent_shape,
            dtype=torch.float16,
            device=self.device,
            generator=self.generator
        )

    def random_mutation(
        self,
        selected_latent: torch.Tensor,
        num_children: int = 4
    ) -> List[torch.Tensor]:
        """ランダム突然変異を適用"""
        new_latents = []
        
        # 選択された潜在ベクトルから新しい画像を生成
        for i in range(1, num_children + 1):
            # ノイズを生成（世代に応じて強度を変える）
            noise = randn_tensor(
                self.latent_shape,
                dtype=torch.float16,
                device=self.device
            ) * (i / num_children) * self.current_sigma

            # 新しい潜在ベクトルを生成
            new_latent = selected_latent + noise

            # 再正規化
            norm_factor = torch.sqrt(torch.tensor(float(torch.prod(torch.tensor(self.latent_shape)))))
            new_latent = new_latent / torch.norm(new_latent) * norm_factor
            
            new_latents.append(new_latent)

        # 突然変異強度を更新
        self.current_sigma *= 0.7

        return new_latents

    async def generate_images(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        base_latent: Optional[torch.Tensor] = None,
        num_images: int = 4,
        generation: int = 0,
        session_id: Optional[str] = None,
    ) -> List[Tuple[Image.Image, torch.Tensor]]:
        
        try:
            images_and_latents = []
            
            # セッション・ステップディレクトリの作成
            if session_id:
                output_dir = os.path.join("get_data", session_id, f"step_{generation}")
                os.makedirs(output_dir, exist_ok=True)
            
            print(f"Starting image generation with prompt: {prompt}")
            
            if base_latent is None:
                for i in range(num_images):
                    print(f"Generating image {i+1}/{num_images}")
                    latent = self.generate_initial_noise(seed=i + generation * num_images)
                    image = self._generate_single_image(prompt, negative_prompt, latent)
                    
                    # セッションIDがある場合は画像を保存
                    if session_id:
                        filepath = os.path.join(output_dir, f"image_{i}.png")
                        image.save(filepath)
                    
                    images_and_latents.append((image, latent))
            else:
                print("Generating base image")
                image = self._generate_single_image(prompt, negative_prompt, base_latent)
                images_and_latents.append((image, base_latent))
                
                print("Generating mutated images")
                new_latents = self.random_mutation(base_latent, num_images - 1)
                for i, latent in enumerate(new_latents):
                    print(f"Generating mutated image {i+1}/{len(new_latents)}")
                    image = self._generate_single_image(prompt, negative_prompt, latent)
                    
                    # セッションIDがある場合は画像を保存
                    if session_id:
                        filepath = os.path.join(output_dir, f"image_{i+1}.png")
                        image.save(filepath)
                    
                    images_and_latents.append((image, latent))
            
            print("Image generation completed successfully")
            return images_and_latents

        except Exception as e:
            print(f"Error in generate_images: {str(e)}")
            raise

    def _generate_single_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        latents: Optional[torch.Tensor] = None
    ) -> Image.Image:
        """単一の画像を生成"""
        try:
            output = self.pipe(
                prompt,
                height=512,
                width=512,
                latents=latents,
                negative_prompt=negative_prompt,
                num_inference_steps=1,
                guidance_scale=0.0
            )
            return output.images[0]

        except Exception as e:
            print(f"Error generating single image: {e}")
            raise

    def save_image(self, image: Image.Image, filepath: str):
        """画像を保存"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath)

    def latent_to_base64(self, latent: torch.Tensor) -> str:
            """潜在ベクトルをbase64文字列に変換"""
            # float16として保存されたnumpy配列に変換
            numpy_data = latent.cpu().numpy().astype(np.float16)
            # バイト列に変換
            bytes_data = numpy_data.tobytes()
            # base64エンコード
            base64_str = base64.b64encode(bytes_data).decode('utf-8')
            return base64_str

    def base64_to_latent(self, base64_str: str) -> torch.Tensor:
            """base64文字列を潜在ベクトルに変換"""
            import numpy as np
            # base64デコード
            latent_bytes = base64.b64decode(base64_str)
            # float16型のnumpy配列として読み込み
            latent_np = np.frombuffer(latent_bytes, dtype=np.float16).copy()  # copyを追加
            # 正しい形状に変形
            latent_np = latent_np.reshape(self.latent_shape)
            # tensorに変換してGPUに転送
            return torch.from_numpy(latent_np).to(device=self.device, dtype=torch.float16)
