import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel

from model.time2vec import Time2Vec
from model.bi_channel_attention import BiChannelAttention


class Model(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, emotion_dim: int, pause_dim: int, heads: int, local_window_num: int, dropout_rate: float, trained_filename: str = None):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.pause_dim = pause_dim
        self.heads = heads
        self.local_window_num = local_window_num
        self.dropout_rate = dropout_rate

        # 感情ベクトル (B, emotion_dim)
        self.speaker0_emotion = None
        self.speaker1_emotion = None

        self.text_encoder = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        L = self.text_encoder.config.num_hidden_layers
        for n, p in self.text_encoder.named_parameters():
            if any(f"encoder.layer.{i}." in n for i in [L-4, L - 3, L - 2, L - 1]):
                p.requires_grad = True

        if (pause_dim > 0):
            # 1x1 の学習可能な time 閾値パラメータ
            self.time_threshold = nn.Parameter(torch.tensor(0.0))
            self.time_encoder = Time2Vec(pause_dim=pause_dim)

        self.global_attention_1 = BiChannelAttention(heads, self.hidden_dim, pause_dim, "global", 0, dropout_rate)
        if (local_window_num > 0):
            self.local_attention_1 = BiChannelAttention(heads, self.hidden_dim, pause_dim, "local", local_window_num, dropout_rate)        

        self.fusion_dim = self.hidden_dim + self.pause_dim*self.heads
        
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
        
        self.decoder = nn.Linear(self.fusion_dim + self.emotion_dim*2, num_classes)

        # 感情ベクトル更新用のGRU
        self.speaker_gru = nn.GRUCell(self.fusion_dim, self.emotion_dim)
        self.listener_gru = nn.GRUCell(self.fusion_dim, self.emotion_dim)
        # self.interaction_gru = nn.GRUCell(self.emotion_dim, self.emotion_dim)   # 聞き手の感情から話し手の感情へ影響を与える用
        
        # test用に学習済みモデルをロード
        if (trained_filename is not None):
            file_path = f"./saved_models/IEMOCAP/{trained_filename}"
            self.load_state_dict(torch.load(file_path))
            print(f"Loaded trained model from {file_path}")


    def init_speaker_emotion(self, batch_size: int):
        device = next(self.parameters()).device
        self.speaker0_emotion = torch.zeros(batch_size, self.emotion_dim, device=device)   # (B, emotion_dim)
        self.speaker1_emotion = torch.zeros(batch_size, self.emotion_dim, device=device)   # (B, emotion_dim)


    def renew_speaker_emotion(self, speaker_t: torch.Tensor, fusion_t: torch.Tensor):
        """
        入力:
        speaker_t: (B)  value 0 or 1
        fusion_t:  (B, hidden_size)  [global+local]
        """
        # speaker_t から mask を作成
        speaker_mask = (speaker_t == 1)     # (B)
        mask = speaker_mask.unsqueeze(-1)   # (B,1)

        # select current speaker and listener emotions per batch
        curr_speaker_emotion = torch.where(mask, self.speaker1_emotion, self.speaker0_emotion)    # (B, emotion_dim)
        curr_listener_emotion = torch.where(mask, self.speaker0_emotion, self.speaker1_emotion)   # (B, emotion_dim)

        # update emotions with GRUCell: input=gru_input, hx=current_emotion  
        next_speaker_emotion = self.speaker_gru(fusion_t, curr_speaker_emotion)                    # (B, emotion_dim)
        next_listener_emotion = self.listener_gru(fusion_t, curr_listener_emotion)                 # (B, emotion_dim)
        # next_speaker_emotion = self.interaction_gru(next_listener_emotion, next_speaker_emotion)   # (B, emotion_dim)

        # 更新
        self.speaker0_emotion = torch.where(mask, next_listener_emotion, next_speaker_emotion)   # (B, emotion_dim)
        self.speaker1_emotion = torch.where(mask, next_speaker_emotion, next_listener_emotion)   # (B, emotion_dim)


    def detach_speaker_emotion(self):
        self.speaker0_emotion = self.speaker0_emotion.detach()
        self.speaker1_emotion = self.speaker1_emotion.detach()


    def forward(self, t: int, input_ids_t: torch.Tensor, time_mask: torch.Tensor, utt_mask_t: torch.Tensor, pause_t: torch.Tensor, cache: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        speaker_t = speakers[:, t].contiguous()   # (B)

        # 感情ベクトル初期化
        if (t == 0):
            batch_size = input_ids_t.size(0)
            self.init_speaker_emotion(batch_size)

        outputs = self.text_encoder(input_ids=input_ids_t, attention_mask=utt_mask_t)   # (B, U_max, 768)
        utterance_t = outputs.last_hidden_state[:, 0, :]                                # (B, 768)

        if (self.pause_dim > 0):
            pause_t = torch.relu(pause_t - self.time_threshold)   # (B)
            pause_t = self.time_encoder(pause_t)                  # (B, pause_dim)
        else:
            pause_t = None

        # 形を整形
        utterance_t_chunks = torch.chunk(utterance_t, self.heads, dim=-1)                                # (head, B, hidden_dim/head)
        if (self.pause_dim > 0):
            per_head_content_t = [torch.cat([chunk, pause_t], dim=-1) for chunk in utterance_t_chunks]   # (head, B, hidden_dim/head + pause_dim)
        else:
            per_head_content_t = [chunk for chunk in utterance_t_chunks]                                 # (head, B, hidden_dim/head)
        content_t = torch.cat(per_head_content_t, dim=-1)                                                # (B, (hidden_dim/head + pause_dim)*heads)  =  (B, hidden_dim + pause_dim*heads)

        global_t = self.global_attention_1(t, content_t, time_mask, cache, speakers)                     # (B, hidden_dim + pause_dim*heads)
        if (self.local_window_num > 0):
            local_t = self.local_attention_1(t, content_t, time_mask, cache, speakers)                   # (B, hidden_dim + pause_dim*heads)

        # 2つのattention特徴のfusion
        if (self.local_window_num > 0):
            fusion_t = torch.cat([global_t, local_t], dim=-1)   # (B, (hidden_dim + pause_dim*heads)*2)
            fusion_t = self.fusion_norm(fusion_t)               # (B, (hidden_dim + pause_dim*heads)*4)
            fusion_t = self.fusion_feed_forward(fusion_t)       # (B, hidden_dim + pause_dim*heads)
        else:
            fusion_t = global_t                                 # (B, hidden_dim + pause_dim*heads)
            fusion_t = self.fusion_norm(fusion_t)               # (B, (hidden_dim + pause_dim*heads)*4)
            fusion_t = self.fusion_feed_forward(fusion_t)       # (B, hidden_dim + pause_dim*heads)       

        # 感情ベクトル
        speaker_mask = (speaker_t == 1)     # (B)
        mask = speaker_mask.unsqueeze(-1)   # (B, 1)
        curr_speaker_emotion = torch.where(mask, self.speaker1_emotion, self.speaker0_emotion)   # (B, emotion_dim)
        curr_listener_emotion = torch.where(mask, self.speaker0_emotion, self.speaker1_emotion)  # (B, emotion_dim)

        # 分類
        logits = self.decoder(torch.cat([fusion_t, curr_speaker_emotion, curr_listener_emotion], dim=-1))               # (B, num_classes)

        # 感情ベクトル更新
        self.renew_speaker_emotion(speaker_t, fusion_t)

        return logits, global_t