import torch
import torch.nn as nn

from model.relative_position import t5_relative_position_bucket


class BiChannelAttention(nn.Module):
    def __init__(self, heads: int, utterance_dim: int, pause_dim: int, attention_type: str, local_window_num: int, dropout_rate: float):
        """
        attention_type: "global" or "local" or "speaker" or "listener"
        """
        super(BiChannelAttention, self).__init__()
        self.heads = heads
        self.utterance_dim = utterance_dim
        self.pause_dim = pause_dim

        self.attention_type = attention_type
        self.local_window_num = local_window_num

        # 各headでの次元
        head_utterance_dim = utterance_dim // heads
        head_pause_dim = pause_dim

        self.position_parameter = self.bias_scale = nn.Parameter(torch.tensor(0.02))
        self.heads_dict = nn.ModuleDict({
            f"head_{i}": SingleHeadBiChannelAttention(head_utterance_dim, head_pause_dim, attention_type, local_window_num, dropout_rate, self.position_parameter)
            for i in range(heads)
        })

        # FFN
        # hidden_size = utterance_dim + pause_dim * heads
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(hidden_size, feedforward_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(feedforward_dim, hidden_size),
        # )

        # self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, t: int, content_t: torch.Tensor, time_mask: torch.Tensor, cache: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        residual = content_t   # (B, D + pause_dim*heads)

        content_t_chunks = torch.chunk(content_t, self.heads, dim=-1)                                         # (head, B, D/head + pause_dim)
        cache_chunks = torch.chunk(cache, self.heads, dim=-1) if cache is not None else [None] * self.heads   # (head, t, B, D/head + pause_dim)

        outputs = []   # (head, B, D/head + pause_dim)
        for i, head in enumerate(self.heads_dict.values()):
            head_out = head(t, content_t_chunks[i], time_mask, cache_chunks[i] if cache is not None else None, speakers)   # (B, D/head + pause_dim)
            outputs.append(head_out)

        # 各headの出力を結合 & 残差接続
        output = torch.cat(outputs, dim=-1)   # (B, D + pause_dim*heads)
        output = output + residual            # (B, D + pause_dim*heads)

        # output = output + self.feed_forward(output)   # (B, D + pause_dim*heads)
        # output = self.layer_norm(output)              # (B, D + pause_dim*heads)

        return output



class SingleHeadBiChannelAttention(nn.Module):
    def __init__(self, head_utterance_dim: int, head_pause_dim: int, attention_type: str, local_window_num: int, dropout_rate: float, position_parameter: nn.Parameter):
        super(SingleHeadBiChannelAttention, self).__init__()
        self.head_utterance_dim = head_utterance_dim
        self.head_pause_dim = head_pause_dim

        self.attention_type = attention_type
        self.local_window_num = local_window_num

        self.utterance_q = nn.Linear(head_utterance_dim, head_utterance_dim)
        self.utterance_k = nn.Linear(head_utterance_dim, head_utterance_dim)
        self.utterance_v = nn.Linear(head_utterance_dim, head_utterance_dim)
        self.pause_q = nn.Linear(head_pause_dim, head_pause_dim)
        self.pause_k = nn.Linear(head_pause_dim, head_pause_dim)
        self.pause_v = nn.Linear(head_pause_dim, head_pause_dim)

        self.position_parameter = position_parameter

        self.attn_dropout = nn.Dropout(dropout_rate)


    def forward(self, t: int, content_t_chunk: torch.Tensor, time_mask: torch.Tensor, cache_t_chunk: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        """
        入力:
            content_t_chunk: (B, D/head + pause_dim)
            time_mask_t:     (B, T_max)
            cache_t_chunk:   (B, t, D/head + pause_dim)
        """
        # cache_t_chunkとcontent_t_chunkをutteranceとpauseで分割, cache_t_chunkにcontent_t_chunkを追加
        utterance_t_chunk, pause_t_chunk = torch.split(content_t_chunk, [self.head_utterance_dim, self.head_pause_dim], dim=-1)                 # (B, D/head), (B, pause_dim)
        if (cache_t_chunk is not None):
            cache_t_utterance_chunk, cache_t_pause_chunk = torch.split(cache_t_chunk, [self.head_utterance_dim, self.head_pause_dim], dim=-1)   # (B, t, D/head), (B, t, pause_dim)
            cache_t_utterance_chunk = torch.cat([cache_t_utterance_chunk, utterance_t_chunk.unsqueeze(1)], dim=1)                               # (B, t+1, D/head)
            cache_t_pause_chunk = torch.cat([cache_t_pause_chunk, pause_t_chunk.unsqueeze(1)], dim=1)                                           # (B, t+1, pause_dim)
        else:
            cache_t_utterance_chunk = utterance_t_chunk.unsqueeze(1)   # (B, 1, D/head)
            cache_t_pause_chunk = pause_t_chunk.unsqueeze(1)           # (B, 1, pause_dim)

        utterance_q = self.utterance_q(utterance_t_chunk)
        pause_q = self.pause_q(pause_t_chunk)
        q_t = torch.cat([utterance_q, pause_q], dim=-1)   # (B, D/head + pause_dim)
        q_t = q_t.unsqueeze(1)                            # (B, 1, D/head + pause_dim)

        utterance_k = self.utterance_k(cache_t_utterance_chunk)
        pause_k = self.pause_k(cache_t_pause_chunk)
        k = torch.cat([utterance_k, pause_k], dim=-1)     # (B, t+1, D/head + pause_dim)
        k= k.transpose(1, 2)                              # (B, D/head + pause_dim, t+1)

        utterance_v = self.utterance_v(cache_t_utterance_chunk)
        pause_v = self.pause_v(cache_t_pause_chunk)
        v = torch.cat([utterance_v, pause_v], dim=-1)     # (B, t+1, D/head + pause_dim)

        # 1✕t attention map
        attention_map = torch.bmm(q_t, k)          # (B, 1, t+1)
        attention_map = attention_map.squeeze(1)   # (B, t+1)
        attention_map /= (self.head_utterance_dim + self.head_pause_dim) ** 0.5

        # 相対位置埋め込みの適用
        position_info = t5_relative_position_bucket(t + 1, num_buckets=32, max_distance=128)
        position_info = position_info.to(device=attention_map.device, dtype=attention_map.dtype)   # (t+1,)
        attention_map = attention_map - self.position_parameter * position_info

        # print("Relative Positions:")
        # print(t5_relative_position_bucket(t+1, num_buckets=32, max_distance=128))

        time_mask = time_mask[:, :t+1]   # (B, t+1)
        # True=>0.0, False=>-inf
        time_mask = time_mask.float()
        time_mask = time_mask.masked_fill(time_mask == 1, 0.0) 
        time_mask = time_mask.masked_fill(time_mask == 0, -1e6)

        attention_map = attention_map + time_mask   # (B, t+1)

        # masking for local attention
        seq_len = t+1
        if (self.attention_type == "global"):
            pass
        elif (self.attention_type == "local"):
            local_mask = torch.zeros((len(attention_map), t+1), dtype=attention_map.dtype, device=attention_map.device)
            if (seq_len > self.local_window_num):
                cutoff = seq_len - self.local_window_num
                local_mask[:, :cutoff] = -1e6
                attention_map = attention_map + local_mask
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        attention_map = torch.softmax(attention_map, dim=-1)   # (B, t+1)

        # dropout
        attention_map = self.attn_dropout(attention_map)

        # attention map * v
        output = torch.bmm(attention_map.unsqueeze(1), v).squeeze(1)   # (B, D/head + pause_dim)

        return output