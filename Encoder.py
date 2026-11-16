import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from Decoder import Decoder


def PositionEncoding(H, W, embedding_dim=256):
    d = embedding_dim // 2  # x、y 各占128维
    positions = []

    for idx in range(H * W):
        y = idx // W
        x = idx % W

        pe_y = torch.zeros(d)
        pe_x = torch.zeros(d)

        for i in range(d // 2):
            div = 10000 ** (2 * i / d)

            pe_y[2 * i] = torch.sin(torch.tensor(y / div))
            pe_y[2 * i + 1] = torch.cos(torch.tensor(y / div))

            pe_x[2 * i] = torch.sin(torch.tensor(x / div))
            pe_x[2 * i + 1] = torch.cos(torch.tensor(x / div))

        positions.append(torch.cat([pe_y, pe_x]))

    return torch.stack(positions).unsqueeze(0)

    pos = PositionEncoding(20, 20)
    print(pos.shape)  # [1, 400, 256]


class DetrModel(torch.nn.Module):
    def __init__(self):
        super(DetrModel, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, 256, kernel_size=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.clshead = nn.Linear(256, 81)
        self.bboxhead = nn.Linear(256, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        # 位置编码与特征图对应：[1,H*W,256]。
        H, W = x.shape[-2:]
        P = PositionEncoding(H, W).to(device=x.device, dtype=x.dtype)
        x = x.flatten(start_dim=2).permute(
            0, 2, 1
        )  # (batch_size, num_patches, embedding_dim)
        x = self.encoder(x, P)
        x = self.decoder(x, P)  # [B,100,256]，每个 Query 输出一组预测。
        cls_logits = self.clshead(x)
        bbox_preds = self.bboxhead(x).sigmoid()  # 归一化的 cxcywh：[B,100,4]。
        return x, cls_logits, bbox_preds


class EncoderAttention(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, d_model),
        )

    def forward(self, X, P):
        B, N, C = X.shape  # [2, 400, 256]

        # 第一步：QKV 线性投影（权重可学习）
        Q = self.Wq(X + P)
        K = self.Wk(X + P)
        V = self.Wv(X)  # detr的公式计算

        # 第二步：拆分 8 个 Head
        Q = Q.reshape(B, N, 8, 32).transpose(1, 2)
        K = K.reshape(B, N, 8, 32).transpose(1, 2)
        V = V.reshape(B, N, 8, 32).transpose(1, 2)
        # 得到 [2, 8, 400, 32]，8个head，每个head的维度是32
        # 第三步：计算注意力权重
        scores = Q @ K.transpose(-2, -1)

        # K.transpose(-2, -1)是将K的最后两个维度进行转置，得到[2, 8, 32, 400]，然后与Q相乘，得到[2, 8, 400, 400]，表示每个head的注意力权重矩阵
        scores = scores / (32**0.5)

        attn = torch.softmax(scores, dim=-1)
        # attn = [2, 8, 400, 400]，表示每个head的注意力权重矩阵，经过softmax归一化后，表示每个query对所有key的注意力分布
        # 第四步：加权聚合 Value
        out = attn @ V
        # [2, 8, 400, 32]

        # 第五步：合并 8 个 Head
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, N, C)
        # [2, 400, 256]

        # 第六步：可学习的输出投影
        out = self.Wo(out)
        # out = [2, 400, 256]
        # Self-Attention 后
        X = self.norm1(X + self.dropout1(out))
        # FFN
        ffn_out = self.ffn(X)

        X = self.norm2(X + self.dropout2(ffn_out))
        return X


class Encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderAttention(d_model, nhead) for _ in range(num_layers)]
        )

    def forward(self, X, P):
        for layer in self.layers:
            X = layer(X, P)

        return X


if __name__ == "__main__":
    encoder = Encoder()

    X = torch.randn(2, 400, 256)
    P = torch.randn(1, 400, 256)  # 仅测试形状，正式使用2D位置编码

    out = encoder(X, P)

    print(out.shape)
    # torch.Size([2, 400, 256])
