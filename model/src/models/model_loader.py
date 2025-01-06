import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging
from ..models.architectures import GazeLSTM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelLoader:
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = Path(model_dir)
        self.device = device
        self.current_model = None
        self.current_model_info = None
        
        logger.info(f"Initialized ModelLoader with model_dir: {model_dir}, device: {device}")

    def list_available_models(self) -> List[Dict[str, Any]]:
        available_models = []
        try:
            # .pthファイルを検索
            for model_path in self.model_dir.glob("*.pth"):
                logger.debug(f"Found model file: {model_path}")
                model_name = model_path.stem
                available_models.append({
                    'name': model_name,
                    'type': 'lstm',
                    'method': 'method1'
                })
                logger.debug(f"Added model to available list: {model_name}")

            logger.info(f"Found {len(available_models)} available models")
            return available_models

        except Exception as e:
            logger.error(f"Error listing available models: {str(e)}")
            raise

    def load_model(self, model_name: str) -> None:
        try:
            logger.info(f"Loading model: {model_name}")
            model_path = self.model_dir / f"{model_name}.pth"

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # 状態辞書を読み込み
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # LSTMの構造を状態辞書から推測
            num_layers = sum(1 for k in state_dict.keys() if 'weight_ih_l' in k)
            hidden_size = state_dict['lstm.weight_ih_l0'].size(0) // 4
            
            logger.info(f"Detected model structure: num_layers={num_layers}, hidden_size={hidden_size}")

            # モデルをインスタンス化
            model = GazeLSTM(
                input_size=4,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=4,
                dropout=0.2
            )

            # 状態辞書をロード
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.current_model = model
            self.current_model_info = {
                'name': model_name,
                'type': 'lstm',
                'method': 'method1',
                'config': {
                    'input_size': 4,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': 0.2
                }
            }
            
            logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    @property
    def is_model_loaded(self) -> bool:
        return self.current_model is not None
