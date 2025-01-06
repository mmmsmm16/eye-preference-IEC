import unittest
import torch
import os
from src.services.stable_diffusion_service import StableDiffusionService

class TestStableDiffusionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = StableDiffusionService()
        cls.test_prompt = "a serene landscape with mountains"
        cls.test_output_dir = "tests/test_outputs"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_initial_noise_generation(self):
        """初期ノイズ生成のテスト"""
        noise = self.service.generate_initial_noise(seed=42)
        self.assertIsInstance(noise, torch.Tensor)
        self.assertEqual(noise.shape, self.service.latent_shape)
        self.assertEqual(noise.device.type, 'cuda')
        self.assertEqual(noise.dtype, torch.float16)

    def test_single_image_generation(self):
        """単一画像生成のテスト"""
        noise = self.service.generate_initial_noise(seed=42)
        image = self.service._generate_single_image(self.test_prompt, None, noise)
        
        # 画像のサイズと形式を確認
        self.assertEqual(image.size, (512, 512))
        
        # テスト用に画像を保存
        test_image_path = os.path.join(self.test_output_dir, "test_single.png")
        self.service.save_image(image, test_image_path)
        self.assertTrue(os.path.exists(test_image_path))

    def test_random_mutation(self):
        """ランダム突然変異のテスト"""
        base_latent = self.service.generate_initial_noise(seed=42)
        mutated_latents = self.service.random_mutation(base_latent, num_children=3)
        
        # 生成された潜在ベクトルの数を確認
        self.assertEqual(len(mutated_latents), 3)
        
        # 各潜在ベクトルの形状とデバイスを確認
        for latent in mutated_latents:
            self.assertEqual(latent.shape, self.service.latent_shape)
            self.assertEqual(latent.device.type, 'cuda')
        
        # 突然変異強度の減少を確認
        initial_sigma = 0.7
        self.assertLess(self.service.current_sigma, initial_sigma)

    def test_full_generation_process(self):
        """完全な生成プロセスのテスト"""
        async def run_test():
            # 初期生成
            results = await self.service.generate_images(
                prompt=self.test_prompt,
                num_images=4,
                generation=0
            )
            
            self.assertEqual(len(results), 4)
            
            # 結果の各要素をテスト
            for i, (image, latent) in enumerate(results):
                self.assertIsNotNone(image)
                self.assertIsNotNone(latent)
                
                # 画像を保存して確認
                test_image_path = os.path.join(self.test_output_dir, f"test_full_{i}.png")
                self.service.save_image(image, test_image_path)
                self.assertTrue(os.path.exists(test_image_path))

            # 突然変異生成
            base_latent = results[0][1]  # 最初の画像の潜在ベクトルを使用
            mutated_results = await self.service.generate_images(
                prompt=self.test_prompt,
                base_latent=base_latent,
                num_images=4,
                generation=1
            )
            
            self.assertEqual(len(mutated_results), 4)

        import asyncio
        asyncio.run(run_test())

    def test_latent_vector_conversion(self):
        """潜在ベクトルの変換テスト"""
        original_latent = self.service.generate_initial_noise(seed=42)
        
        # base64に変換
        base64_str = self.service.latent_to_base64(original_latent)
        self.assertIsInstance(base64_str, str)
        
        # base64から戻す
        recovered_latent = self.service.base64_to_latent(base64_str)
        
        # 元のテンソルと一致することを確認
        torch.testing.assert_close(original_latent.cpu(), recovered_latent.cpu())

if __name__ == '__main__':
    unittest.main()
