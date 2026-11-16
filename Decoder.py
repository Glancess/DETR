import torch
import torch.nn as nn


class Decoder_Self_Attention(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, D, Eobj):
        B, N, C = D.shape  # [2, 400, 256]

        # 第一步：QKV 线性投影（权重可学习）
        Q = self.Wq(D + Eobj)
        K = self.Wk(D + Eobj)
        V = self.Wv(D)  # detr的公式计算

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
        return out


class Decoder_Cross_Attention(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, D, Eobj, M, P):
        B, N, C = D.shape  # [2, 400, 256]
        Nk = M.shape[1]
        # 第一步：QKV 线性投影（权重可学习）
        Q = self.Wq(D + Eobj)
        K = self.Wk(M + P)
        V = self.Wv(M)  # detr的公式计算

        # 第二步：拆分 8 个 Head
        Q = Q.reshape(B, N, 8, 32).transpose(1, 2)
        K = K.reshape(B, Nk, 8, 32).transpose(1, 2)
        V = V.reshape(B, Nk, 8, 32).transpose(1, 2)
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
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = Decoder_Self_Attention(d_model, nhead)
        self.cross_attn = Decoder_Cross_Attention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, d_model),
        )

    def forward(self, D, M, E_obj, P):

        out1 = self.self_attn(D, E_obj)
        D = self.norm1(D + self.dropout1(out1))

        out2 = self.cross_attn(D, E_obj, M, P)
        D = self.norm2(D + self.dropout2(out2))

        ffn_out = self.ffn(D)
        D = self.norm3(D + self.dropout3(ffn_out))

        return D


class Decoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_queries=100):
        super().__init__()

        # 100 个可学习 Object Query Embedding
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 6 层独立的 Decoder Layer
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead) for _ in range(num_layers)]
        )

    def forward(self, M, P):
        B = M.shape[0]

        E_obj = self.query_embed.weight
        # [100,256]

        # Decoder 初始隐藏状态
        D = torch.zeros(
            B, E_obj.shape[0], E_obj.shape[1], device=M.device, dtype=M.dtype
        )

        # 依次通过 6 层 Decoder
        for layer in self.layers:
            D = layer(D, M, E_obj, P)

        return D


if __name__ == "__main__":
    decoder = Decoder()
    M = torch.randn(2, 400, 256)
    P = torch.randn(1, 400, 256)
    D = decoder(M, P)
    print(D.shape)  # [2, 100, 256]
