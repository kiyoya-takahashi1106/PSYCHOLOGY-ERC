import torch
import torch.nn as nn

from model.relative_position import t5_relative_position_bucket


class BiChannelAttention(nn.Module):
    def __init__(self, heads: int, utterance_dim: int, time_dim: int, attention_type: str, local_window_num: int, dropout_rate: float):
        """
        attention_type: "global" or "local" or "speaker" or "listener"
        """
        super(BiChannelAttention, self).__init__()
        self.heads = heads
        self.utterance_dim = utterance_dim
        self.time_dim = time_dim

        self.attention_type = attention_type
        self.local_window_num = local_window_num

        # 各headでの次元
        head_utterance_dim = utterance_dim // heads
        head_time_dim = time_dim

        self.position_parameter = self.bias_scale = nn.Parameter(torch.tensor(0.02))
        self.heads_dict = nn.ModuleDict({
            f"head_{i}": SingleHeadBiChannelAttention(head_utterance_dim, head_time_dim, attention_type, local_window_num, dropout_rate, self.position_parameter)
            for i in range(heads)
        })


    def forward(self, t: int, content_t: torch.Tensor, time_mask: torch.Tensor, cache: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        residual = content_t   # (B, D + time_dim*heads)

        content_t_chunks = torch.chunk(content_t, self.heads, dim=-1)                                         # (head, B, D/head + time_dim)
        cache_chunks = torch.chunk(cache, self.heads, dim=-1) if cache is not None else [None] * self.heads   # (head, t, B, D/head + time_dim)

        outputs = []   # (head, B, D/head + time_dim)
        for i, head in enumerate(self.heads_dict.values()):
            head_out = head(t, content_t_chunks[i], time_mask, cache_chunks[i] if cache is not None else None, speakers)   # (B, D/head + time_dim)
            outputs.append(head_out)

        # 各headの出力を結合 & 残差接続
        output = torch.cat(outputs, dim=-1)   # (B, D + time_dim*heads)
        output = output + residual            # (B, D + time_dim*heads)

        # output = output + self.feed_forward(output)   # (B, D + time_dim*heads)
        # output = self.layer_norm(output)              # (B, D + time_dim*heads)

        return output



class SingleHeadBiChannelAttention(nn.Module):
    def __init__(self, head_utterance_dim: int, head_time_dim: int,
                 attention_type: str, local_window_num: int,
                 dropout_rate: float, position_parameter: nn.Parameter):
        super(SingleHeadBiChannelAttention, self).__init__()
        self.head_utterance_dim = head_utterance_dim
        self.head_time_dim = head_time_dim
        self.head_dim = head_utterance_dim + head_time_dim

        self.attention_type = attention_type
        self.local_window_num = local_window_num

        # ★ ここを「共通の Wq, Wk, Wv」に変更 ★
        self.q_proj = nn.Linear(self.head_dim, self.head_dim)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim)

        self.position_parameter = position_parameter
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, t: int, content_t_chunk: torch.Tensor,
                time_mask: torch.Tensor, cache_t_chunk: torch.Tensor,
                speakers: torch.Tensor) -> torch.Tensor:
        """
        入力:
            content_t_chunk: (B, D/head + time_dim)
            time_mask:       (B, T_max)
            cache_t_chunk:   (B, t, D/head + time_dim) or None
        """
        # cache に現在のステップ t のトークンを追加
        if cache_t_chunk is not None:
            # (B, t+1, D/head + time_dim)
            cache_full = torch.cat(
                [cache_t_chunk, content_t_chunk.unsqueeze(1)],
                dim=1
            )
        else:
            cache_full = content_t_chunk.unsqueeze(1)  # (B, 1, D/head + time_dim)

        # ★ 共通の Wq, Wk, Wv を適用（普通の Attention と同じイメージ）★
        q_t = self.q_proj(content_t_chunk)         # (B, D/head + time_dim)
        q_t = q_t.unsqueeze(1)                     # (B, 1, D/head + time_dim)

        k = self.k_proj(cache_full)                # (B, t+1, D/head + time_dim)
        k = k.transpose(1, 2)                      # (B, D/head + time_dim, t+1)

        v = self.v_proj(cache_full)                # (B, t+1, D/head + time_dim)

        # 1✕(t+1) attention map
        attention_map = torch.bmm(q_t, k)          # (B, 1, t+1)
        attention_map = attention_map.squeeze(1)   # (B, t+1)
        attention_map /= (self.head_dim ** 0.5)

        # 相対位置埋め込みの適用
        position_info = t5_relative_position_bucket(
            t + 1, num_buckets=32, max_distance=128
        )
        position_info = position_info.to(
            device=attention_map.device,
            dtype=attention_map.dtype
        )  # (t+1,)
        attention_map = attention_map - self.position_parameter * position_info

        # time mask 適用
        time_mask = time_mask[:, :t+1]   # (B, t+1)
        time_mask = time_mask.float()
        # True -> 0.0, False -> -inf という前提のまま
        time_mask = time_mask.masked_fill(time_mask == 1, 0.0)
        time_mask = time_mask.masked_fill(time_mask == 0, -1e6)

        attention_map = attention_map + time_mask  # (B, t+1)

        # local / global attention の切り替え
        seq_len = t + 1
        if self.attention_type == "global":
            pass
        elif self.attention_type == "local":
            local_mask = torch.zeros_like(attention_map)
            if seq_len > self.local_window_num:
                cutoff = seq_len - self.local_window_num
                local_mask[:, :cutoff] = -1e6
                attention_map = attention_map + local_mask
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        # softmax & dropout
        attention_map = torch.softmax(attention_map, dim=-1)  # (B, t+1)
        attention_map = self.attn_dropout(attention_map)

        # attention map * v
        output = torch.bmm(attention_map.unsqueeze(1), v).squeeze(1)
        # (B, D/head + time_dim)

        return output