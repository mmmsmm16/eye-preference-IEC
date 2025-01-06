from fastapi.testclient import TestClient
import unittest
from src.api.server import app
import os
import json
from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image

class TestGenerationAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_prompt = "a serene landscape with mountains"
        
        # テスト用の画像ディレクトリとファイルを作成
        cls.test_image_dir = "test_get_data"
        os.makedirs(cls.test_image_dir, exist_ok=True)
        
        # テスト用のダミー画像を作成
        cls.test_image_path = os.path.join(cls.test_image_dir, "test_image.png")
        if not os.path.exists(cls.test_image_path):
            # PIL を使用してダミー画像を作成
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(cls.test_image_path)

    def test_initial_generation(self):
        """初期画像生成APIのテスト"""
        response = self.client.post(
            "/generate",  # URLパスを修正
            json={
                "prompt": self.test_prompt,
                "num_images": 4,
                "generation": 0
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("images", data)
        self.assertEqual(len(data["images"]), 4)
        self.assertEqual(data["generation"], 0)
        self.assertEqual(data["base_prompt"], self.test_prompt)

    def test_mutation_generation(self):
        """突然変異による画像生成APIのテスト"""
        initial_response = self.client.post(
            "/generate",  # URLパスを修正
            json={
                "prompt": self.test_prompt,
                "num_images": 4,
                "generation": 0
            }
        )
        
        initial_data = initial_response.json()
        base_vector = initial_data["images"][0]["latent_vector"]

        mutation_response = self.client.post(
            "/generate",  # URLパスを修正
            json={
                "prompt": self.test_prompt,
                "base_vector": base_vector,
                "num_images": 4,
                "generation": 1
            }
        )
        
        self.assertEqual(mutation_response.status_code, 200)
        mutation_data = mutation_response.json()
        
        self.assertEqual(len(mutation_data["images"]), 4)
        self.assertEqual(mutation_data["generation"], 1)

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        response = self.client.post(
            "/generate",  # URLパスを修正
            json={
                "prompt": "",  # 空のプロンプト
                "num_images": 4,
                "generation": 0
            }
        )
        self.assertEqual(response.status_code, 422)

    def test_image_retrieval(self):
        """画像取得APIのテスト"""
        # まず画像生成
        generation_response = self.client.post(
            "/generate",
            json={
                "prompt": self.test_prompt,
                "num_images": 1,
                "generation": 0
            }
        )
        
        self.assertEqual(generation_response.status_code, 200)
        data = generation_response.json()
        
        # 生成された画像のURLを取得
        image_url = data["images"][0]["url"]
        
        # 画像の取得をテスト
        image_response = self.client.get(image_url)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.headers["content-type"], "image/png")

    @classmethod
    def tearDownClass(cls):
        # テスト終了後のクリーンアップ
        if os.path.exists(cls.test_image_dir):
            import shutil
            shutil.rmtree(cls.test_image_dir)

if __name__ == '__main__':
    unittest.main()
