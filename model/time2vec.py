import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, pause_dim: int):
        super(Time2Vec, self).__init__()
        self.pause_dim = pause_dim

        self.weight_linear = nn.Parameter(torch.randn(1, 1))
        self.bias_linear = nn.Parameter(torch.randn(1, 1))

        self.weight_periodic = nn.Parameter(torch.randn(1, pause_dim - 1))
        self.bias_periodic = nn.Parameter(torch.randn(1, pause_dim - 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力:
            x: (B)  # 時間情報
        出力:
            time2vec_output: (B, pause_dim)
        """
        x = x.unsqueeze(-1)

        linear_part = torch.matmul(x, self.weight_linear) + self.bias_linear
        periodic_part = torch.sin(torch.matmul(x, self.weight_periodic) + self.bias_periodic)

        return torch.cat([linear_part, periodic_part], dim=-1)