import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm  
from config import Config

# ----------------------------
# 路徑與資料載入
# ----------------------------
base_dir = Config.SPLIT_DATA_DIR
trainX_path = os.path.join(base_dir, "trainX.npy")
trainY_path = os.path.join(base_dir, "trainY.npy")
valX_path = os.path.join(base_dir, "valX.npy")
valY_path = os.path.join(base_dir, "valY.npy")
testX_path = os.path.join(base_dir, "testX.npy")
testY_path = os.path.join(base_dir, "testY.npy")

def load_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_train = torch.tensor(np.load(trainX_path), dtype=torch.float32)
    y_train = torch.tensor(np.load(trainY_path), dtype=torch.long)
    X_val   = torch.tensor(np.load(valX_path),   dtype=torch.float32)
    y_val   = torch.tensor(np.load(valY_path),   dtype=torch.long)
    X_test  = torch.tensor(np.load(testX_path),  dtype=torch.float32)
    y_test  = torch.tensor(np.load(testY_path),  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


# ----------------------------
# 模組：SE Block
# ----------------------------
class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        hidden = max(1, channel // reduction)
        self.fc1 = nn.Linear(channel, hidden)
        self.fc2 = nn.Linear(hidden, channel)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.size()
        y = self.gap(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y


# ----------------------------
# 模組：MPCA Block（論文版本）
# 三支卷積 → concat(192) → SE → 1×1 Conv(192→64)
# Simple path 使用兩層point-wise conv，作為殘差相加
# ----------------------------
class MPCA_Block(nn.Module):
    def __init__(self, in_channels: int = 4, seq_len: int = 1024):
        super().__init__()
        from config import Config
        d_model = Config.D_MODEL  # 64

        # Branch 1 (Conv15→MaxPool→Conv9→MaxPool)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 新增
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 新增
        )
        # Branch 2 (Conv13→MaxPool→Conv7→MaxPool)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=13, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Branch 3 (Conv11→MaxPool→Conv5→MaxPool)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        concat_channels = 3 * 64
        self.se = SEBlock(channel=concat_channels, reduction=16)
        self.concat_conv = nn.Conv1d(concat_channels, d_model, kernel_size=1)

        self.simple_path = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 新增
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 新增
        )

    def forward(self, x):
        x1 = self.branch1(x)  # [B, 64, 128]
        x2 = self.branch2(x)  # [B, 64, 128]
        x3 = self.branch3(x)  # [B, 64, 128]

        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 192, 128]
        x_se = self.se(x_cat)
        x_se = self.concat_conv(x_se)           # [B, 64, 128]

        x_simple = self.simple_path(x)          # [B, 64, 128]

        return x_se + x_simple                  # [B, 64, 128]


# ----------------------------
# 模組：Positional Encoding（支援 1024）
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ----------------------------
# Transformer Encoder Layer（論文設定）
# d_model=64, nhead=4, FFN hidden dim=512
# ----------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn_q = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_k = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_v = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_out = nn.Linear(d_model, d_model, bias=True)

        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model == self.head_dim * nhead

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=True),
        )

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.self_attn_q, self.self_attn_k, self.self_attn_v, self.self_attn_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        B, T, D = x.size()

        # Pre-norm
        residual = x
        x = self.norm1(x)

        Q = self.self_attn_q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.self_attn_k(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.self_attn_v(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.self_attn_out(attn_output)

        x = residual + self.dropout1(attn_output)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)

        return x


# ----------------------------
# 模型本體（論文版本）
# ----------------------------
class HybridOSA_Model(nn.Module):
    def __init__(self, seq_len: int = 1024, in_channels: int = 4, num_classes: int = 2):
        super().__init__()
        from config import Config
        self.seq_len = seq_len

        self.mpca = MPCA_Block(in_channels=in_channels, seq_len=seq_len)  # [B, 64, 1024]
        self.pos_enc = PositionalEncoding(d_model=Config.D_MODEL, max_len=seq_len)

        self.transformer_layers = nn.Sequential(*[
            TransformerEncoderLayer(
                d_model=Config.D_MODEL,
                nhead=Config.N_HEADS,
                dim_feedforward=512,   # 論文設定
                dropout=Config.DROPOUT
            ) for _ in range(3)
        ])

        # point-wise feedforward (64→64)
        self.compress = nn.Linear(Config.D_MODEL, Config.D_MODEL)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 分類頭 (64→2)
        self.fc = nn.Linear(Config.D_MODEL, num_classes)

    def forward(self, x):
        # x: [B, 1024, 4]
        x = x.permute(0, 2, 1)        # [B, 4, 1024]
        x = self.mpca(x)              # [B, 64, 128]

        x = x.permute(0, 2, 1)        # [B, 128, 64]
        x = self.pos_enc(x)           # [B, 128, 64]
        x = self.transformer_layers(x)# [B, 128, 64]

        x = self.compress(x)          # [B, 128, 64]

        x = x.permute(0, 2, 1)        # [B, 64, 128]
        x = self.global_avg_pool(x)   # [B, 64, 1]
        x = x.squeeze(-1)             # [B, 64]

        x = self.fc(x)                # [B, 2]
        return x


# ----------------------------
# Focal Loss（類別不平衡）
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 若給 tensor([alpha_0, alpha_1]) 會做 class weighting
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C], targets: [B]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# 計算類別權重
def calculate_class_weights_auto(train_loader, device):
    """
    基於類別比例自動計算權重
    """
    class_counts = torch.zeros(2)
    for _, targets in train_loader:
        for class_idx in range(2):
            class_counts[class_idx] += (targets == class_idx).sum().item()
    
    # 反比例權重
    total_samples = class_counts.sum()
    weights = total_samples / (2 * class_counts)
    
    print(f"📊 自動計算權重:")
    print(f"  類別 0 (正常): {int(class_counts[0])} 樣本 → 權重 {weights[0]:.3f}")
    print(f"  類別 1 (呼吸中止): {int(class_counts[1])} 樣本 → 權重 {weights[1]:.3f}")
    print(f"  類別比例: 1:{class_counts[0]/class_counts[1]:.2f}")
    
    return weights.to(device)


# ----------------------------
# 參數統計
# ----------------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
