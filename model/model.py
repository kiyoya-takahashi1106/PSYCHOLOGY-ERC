import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from transformers import RobertaModel

from model.time2vec import Time2Vec
from model.bi_channel_attention import BiChannelAttention


class Model(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, speaker_state_dim: int, time_dim: int, heads: int, local_window_num: int, dropout_rate: float, trained_filename: str = None):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.speaker_state_dim = speaker_state_dim
        self.time_dim = time_dim
        self.heads = heads
        self.local_window_num = local_window_num
        self.dropout_rate = dropout_rate
        self.interaction_heads = 6

        # speaker状態 (B, emotion_dim), (B, interaction_heads)
        self.speaker0_state = None
        self.speaker1_state = None
        self.interaction_state = None  


        # ====== ここから Interaction 用 ======
        self.head_interaction_dim = self.speaker_state_dim // self.interaction_heads

        # heads 次元 (B, H) → (B, interaction_dim)
        self.interaction_dim = self.speaker_state_dim // 2
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.interaction_heads, self.interaction_dim),
        )
        # ======  Interaction ここまで  ======


        self.text_encoder = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        L = self.text_encoder.config.num_hidden_layers
        for n, p in self.text_encoder.named_parameters():
            if any(f"encoder.layer.{i}." in n for i in [L-4, L - 3, L - 2, L - 1]):
                p.requires_grad = True

        if (time_dim > 0):
            # 1x1 の学習可能な time 閾値パラメータ
            # self.time_threshold = nn.Parameter(torch.tensor(0.0))
            self.time_encoder = Time2Vec(time_dim=time_dim)

        self.global_attention_1 = BiChannelAttention(heads, self.hidden_dim, time_dim, "global", 0, dropout_rate)
        if (local_window_num > 0):
            self.local_attention_1 = BiChannelAttention(heads, self.hidden_dim, time_dim, "local", local_window_num, dropout_rate)        

        self.fusion_dim = self.hidden_dim + self.time_dim*self.heads
        
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        if (local_window_num > 0):
            self.fusion_feed_forward = nn.Sequential(
                nn.Linear(self.fusion_dim*2, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.fusion_dim, self.fusion_dim),
            )
        else:
            self.fusion_feed_forward = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.fusion_dim, self.fusion_dim),
            )
        
        self.decoder = nn.Linear(self.fusion_dim + self.speaker_state_dim + self.interaction_dim, num_classes)

        # speaker状態 更新用のGRU
        self.speaker_gru = nn.GRUCell(self.fusion_dim, self.speaker_state_dim)
        self.interaction_gru = nn.GRUCell(self.interaction_heads, self.interaction_heads)
        
        # test用に学習済みモデルをロード
        if (trained_filename is not None):
            file_path = f"./saved_models/IEMOCAP/{trained_filename}"
            self.load_state_dict(torch.load(file_path))
            print(f"Loaded trained model from {file_path}")


    def init_speaker_state(self, batch_size: int):
        device = next(self.parameters()).device
        self.speaker0_state = torch.zeros(batch_size, self.speaker_state_dim, device=device)   # (B, emotion_dim)
        self.speaker1_state = torch.zeros(batch_size, self.speaker_state_dim, device=device)   # (B, emotion_dim)
        self.interaction_state = torch.zeros(batch_size, self.interaction_heads, device=device)  # (B, interaction_heads)


    def renew_speaker_state(self, speaker_t: torch.Tensor, fusion_t: torch.Tensor):
        """
        入力:
        speaker_t: (B)  value 0 or 1
        fusion_t:  (B, hidden_size)  [global+local]
        """
        # speaker_t から mask を作成
        speaker_mask = (speaker_t == 1)     # (B)
        mask = speaker_mask.unsqueeze(-1)   # (B,1)

        # select current speaker and listener emotions per batch
        curr_speaker_state = torch.where(mask, self.speaker1_state, self.speaker0_state)    # (B, emotion_dim)
        curr_listener_state = torch.where(mask, self.speaker0_state, self.speaker1_state)   # (B, emotion_dim)

        # update state with GRUCell: input=gru_input, hx=current_emotion  
        next_speaker_state = self.speaker_gru(fusion_t, curr_speaker_state)                    # (B, emotion_dim)
        # next_listener_state = self.listener_gru(fusion_t, curr_listener_state)                 # (B, emotion_dim)

        # 更新
        self.speaker0_state = torch.where(mask, curr_listener_state, next_speaker_state)   # (B, emotion_dim)
        self.speaker1_state = torch.where(mask, next_speaker_state, curr_listener_state)   # (B, emotion_dim)


    def detach_state(self):
        self.speaker0_state = self.speaker0_state.detach()
        self.speaker1_state = self.speaker1_state.detach()
        self.interaction_state = self.interaction_state.detach()


    def cul_interaction(self, curr_speaker_state: torch.Tensor, curr_listener_state: torch.Tensor) -> torch.Tensor:
        """
        入力:
        curr_speaker_state:  (B, emotion_dim)
        curr_listener_state: (B, emotion_dim)
        出力:
        interaction:         (B, interaction_dim)
        """
        B, _ = curr_speaker_state.size()

        # (B, D) → (B, interacton_heads * head_interaction_dim)
        # curr_speaker_state = self.interaction_speaker_linear(curr_speaker_state)
        # curr_listener_state = self.interaction_speaker_linear(curr_listener_state)
        curr_speaker_state = torch.stack(torch.chunk(curr_speaker_state, self.interaction_heads, dim=-1), dim=1)
        curr_listener_state = torch.stack(torch.chunk(curr_listener_state, self.interaction_heads, dim=-1), dim=1)

        # (B, interacton_heads * head_interaction_dim) → (B, interacton_heads, head_interaction_dim)
        curr_speaker_state = curr_speaker_state.view(B, self.interaction_heads, self.head_interaction_dim)
        curr_listener_state = curr_listener_state.view(B, self.interaction_heads, self.head_interaction_dim)

        # 各ヘッドごとに内積 → (B, interacton_heads)
        relations = (curr_speaker_state * curr_listener_state).sum(dim=-1) / math.sqrt(self.head_interaction_dim)
        relations = torch.tanh(relations)

        # gruでinteraction状態更新
        next_interaction_state = self.interaction_gru(relations, self.interaction_state)   # (B, interaction_heads)
        self.interaction_state = next_interaction_state

        # MLP で (B, interaction_dim) アップサンプリング
        interaction = self.interaction_mlp(self.interaction_state)   # (B, interaction_dim)

        return interaction


    def forward(self, t: int, input_ids_t: torch.Tensor, time_mask: torch.Tensor, utt_mask_t: torch.Tensor, speed_t: torch.Tensor, pause_t: torch.Tensor, cache: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        speaker_t = speakers[:, t].contiguous()   # (B)

        # speaker状態初期化
        if (t == 0):
            batch_size = input_ids_t.size(0)
            self.init_speaker_state(batch_size)

        outputs = self.text_encoder(input_ids=input_ids_t, attention_mask=utt_mask_t)   # (B, U_max, 768)
        utterance_t = outputs.last_hidden_state[:, 0, :]                                # (B, 768)

        if (self.time_dim > 0):
            # speed_t = torch.relu(speed_t - self.time_threshold)   # (B)
            speed_t = self.time_encoder(speed_t)                    # (B, time_dim)
        else:
            speed_t = None

        # 形を整形
        utterance_t_chunks = torch.chunk(utterance_t, self.heads, dim=-1)                                # (head, B, hidden_dim/head)
        if (self.time_dim > 0):
            per_head_content_t = [torch.cat([chunk, speed_t], dim=-1) for chunk in utterance_t_chunks]   # (head, B, hidden_dim/head + time_dim)
        else:
            per_head_content_t = [chunk for chunk in utterance_t_chunks]                                 # (head, B, hidden_dim/head)
        content_t = torch.cat(per_head_content_t, dim=-1)                                                # (B, (hidden_dim/head + time_dim)*heads)  =  (B, hidden_dim + time_dim*heads)

        global_t = self.global_attention_1(t, content_t, time_mask, cache, speakers)                     # (B, hidden_dim + time_dim*heads)
        if (self.local_window_num > 0):
            local_t = self.local_attention_1(t, content_t, time_mask, cache, speakers)                   # (B, hidden_dim + time_dim*heads)

        # 2つのattention特徴のfusion
        if (self.local_window_num > 0):
            fusion_t = torch.cat([global_t, local_t], dim=-1)   # (B, (hidden_dim + time_dim*heads)*2)
            fusion_t = self.fusion_norm(fusion_t)               # (B, (hidden_dim + time_dim*heads)*4)
            fusion_t = self.fusion_feed_forward(fusion_t)       # (B, hidden_dim + time_dim*heads)
        else:
            fusion_t = global_t                                 # (B, hidden_dim + time_dim*heads)
            fusion_t = self.fusion_norm(fusion_t)               # (B, (hidden_dim + time_dim*heads)*4)
            fusion_t = self.fusion_feed_forward(fusion_t)       # (B, hidden_dim + time_dim*heads)       

        # speaker状態更新
        self.renew_speaker_state(speaker_t, fusion_t)
        
        # speaker状態
        speaker_mask = (speaker_t == 1)     # (B)
        mask = speaker_mask.unsqueeze(-1)   # (B, 1)
        curr_speaker_state = torch.where(mask, self.speaker1_state, self.speaker0_state)   # (B, emotion_dim)
        curr_listener_state = torch.where(mask, self.speaker0_state, self.speaker1_state)  # (B, emotion_dim)

        interaction = self.cul_interaction(curr_speaker_state, curr_listener_state) 

        # 分類
        logits = self.decoder(torch.cat([fusion_t, curr_speaker_state, interaction], dim=-1))               # (B, num_classes)

        return logits, global_t