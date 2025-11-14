import torch


class CollateFn:
    def __init__(self, pad_id: int = 1, label_pad_value: int = -100, pad_speaker_id: int = 0):
        self.pad_id = pad_id
        self.label_pad_value = label_pad_value
        self.speaker2id = {"Male": 0, "Female": 1}
        self.pad_speaker_id = pad_speaker_id


    def __call__(self, batch):

        """
            batch: list[conv], 
            conv: list[{"input_ids": List[int], "pause": float, "label": int}]
            
            出力:
            input_ids:      (B, T_max, U_max)  long
            time_mask:      (B, T_max)         bool  (True=valid)
            utt_mask:       (B, T_max, U_max)  bool  (True=valid)
            pauses:         (B, T_max)         float
            speakers:       (B, T_max)         long
            labels:         (B, T_max)         long  (-100 は無視)
        """

        B = len(batch)
        T_max = max(len(conv) for conv in batch)                               # 最大発話数 (ステップ数)
        U_max = max(len(utt["input_ids"]) for conv in batch for utt in conv)   # 最大トークン長

        # init
        input_ids   = torch.full((B, T_max, U_max), self.pad_id, dtype=torch.long)
        time_mask   = torch.zeros((B, T_max), dtype=torch.bool)
        utt_mask    = torch.zeros((B, T_max, U_max), dtype=torch.bool)
        pauses      = torch.zeros((B, T_max), dtype=torch.float)
        speakers    = torch.full((B, T_max), self.pad_speaker_id, dtype=torch.long)
        labels      = torch.full((B, T_max), self.label_pad_value, dtype=torch.long)

        for i, conv in enumerate(batch):
            for j, utt in enumerate(conv):
                ids = utt["input_ids"]
                L = len(ids)
                if (L > 0):
                    input_ids[i, j, :L] = torch.tensor(ids[:L])
                    utt_mask[i, j, :L] = True
                time_mask[i, j] = True
                pauses[i, j] = float(utt["pause"])
                speakers[i, j] = self.speaker2id[str(utt["speaker"])]
                labels[i, j] = int(utt["label"])

        return {
            "input_ids": input_ids,             # (B, T_max, U_max)
            "time_mask": time_mask,             # (B, T_max)
            "utt_mask": utt_mask,               # (B, T_max, U_max)
            "pauses": pauses,                   # (B, T_max)
            "speakers": speakers,               # (B, T_max)
            "labels": labels,                   # (B, T_max)
        }