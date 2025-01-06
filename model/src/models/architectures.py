import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].transpose(0, 1)

class GazeLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=4, num_classes=4, dropout=0.2):
        super(GazeLSTM, self).__init__()
        
        # LSTMレイヤーの設定
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        # Forward propagate LSTM
        output, (hidden, cell) = self.lstm(x)
        
        # Use dropout
        output = self.dropout(output)
        
        # Get the outputs from the last non-padded elements
        if lengths is not None:
            batch_size = output.size(0)
            last_outputs = output[torch.arange(batch_size), lengths - 1]
        else:
            last_outputs = output[:, -1]
        
        # Pass through the fully connected layer
        logits = self.fc(last_outputs)
        return logits

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class GazeTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, nhead=8, dropout=0.2):
        super(GazeTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 入力の線形変換
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformerエンコーダー層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 出力の線形変換
        self.fc = nn.Linear(hidden_size, num_classes)

    def create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        mask = torch.arange(max_len).expand(len(lengths), max_len).to(lengths.device)
        mask = mask < lengths.unsqueeze(1)
        return ~mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor = None):
        # 入力を隠れ層の次元に変換
        x = self.input_proj(x)
        
        # 位置エンコーディングを追加
        x = self.pos_encoder(x)
        
        # パディングマスクを生成（lengthsが提供された場合）
        if lengths is not None:
            padding_mask = self.create_mask(lengths, x.size(1)).to(x.device)
            # Transformerエンコーダーを適用（マスクあり）
            encoder_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            # Transformerエンコーダーを適用（マスクなし）
            encoder_output = self.transformer_encoder(x)
        
        # 各シーケンスの最後の有効な出力を取得
        if lengths is not None:
            batch_size = encoder_output.size(0)
            last_outputs = encoder_output[torch.arange(batch_size), lengths - 1]
        else:
            last_outputs = encoder_output.mean(dim=1)  # 全時刻の平均を取る
        
        # ドロップアウトを適用
        last_outputs = self.dropout(last_outputs)
        
        # 分類層を適用
        logits = self.fc(last_outputs)
        
        return logits
