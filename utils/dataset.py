import torch

from transformers import AutoTokenizer

import json


"""
    data_dct = {
        # 会話1
        Ses05F_impro01: ["Ses05F_impro01_F000", "Ses05F_impro01_M000", ...],
        # 会話2
        Ses05F_script02_1: ["Ses05F_script02_1_M000", "Ses05F_script02_1_M001", ...],
        ...
    }

    data = [
        # 会話1
        [ 
            # 発話1
            {"input_ids": [tokenizeから返ってくるid], "pause": pause(秒数), "speaker": sex, "label": label},
            # 発話2
            {"input_ids": [tokenizeから返ってくるid], "pause": pause(秒数), "speaker": sex, "label": label},
            ...
        ], 
        ...
    ]
"""


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

        self.basic_dir_path = f"./data/{dataset}"
        self.data_dir_path = f"{self.basic_dir_path}/raw-texts/{split}"
        
        self.speaker_dct = {
            "Ses01": {"Female": "Mary", "Male": "James"},
            "Ses02": {"Female": "Patricia", "Male": "John"},
            "Ses03": {"Female": "Jennifer", "Male": "Robert"},
            "Ses04": {"Female": "Linda", "Male": "Michael"},
            "Ses05": {"Female": "Elizabeth", "Male": "William"},
        }
        self.max_len = 128
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.ignore_label = -100
        self.emotion2label = {"happiness":0, "sadness":1, "neutral":2, "anger":3, "excited":4, "frustration":5}

        self.data_dct = self.load_data_dct()
        self.data = self.load_data()


    def load_data_dct(self):
        file = f"{self.basic_dir_path}/utterance-ordered.json"
        with open(file, "r", encoding="utf-8") as f:
            data_dct = json.load(f)
        data_dct = data_dct[self.split]

        return data_dct
    

    def load_data(self):
        data = []   # 全会話数 ✕ その会話の発話数

        for _, utterance_lst in self.data_dct.items():
            conv_data = []   # 1会話分の発話数
            # prev_end_time = 0.0

            for i, utterance_file in enumerate(utterance_lst):
                file_path = f"{self.data_dir_path}/{utterance_file}.json"

                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = json.load(f)

                utterance = file_content["Utterance"]
                start_time = file_content["StartTime"]
                end_time = file_content["EndTime"]
                sex = file_content["Speaker"]
                emotion = file_content["Emotion"]

                # 発言内容 => "speaker_name": 発言内容
                session_id = utterance_file.split("_")[0][:5]
                speaker_name = self.speaker_dct[session_id][sex]
                name_utterance = speaker_name + ": " + utterance
    
                tokenizer_output = self.tokenizer(
                    name_utterance,
                    padding=False,
                    truncation=True,
                    max_length=self.max_len,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )

                # if (i == 0):
                #     pause = 0.0
                # else:
                #     pause = start_time - prev_end_time
                # prev_end_time = end_time

                char_num = 0
                for word in utterance.split():
                    char_num += len(word)
                    
                speed = (end_time - start_time) / char_num

                if (emotion in self.emotion2label):
                    label = self.emotion2label[emotion]
                else:
                    label = self.ignore_label   # 無視ラベル

                # conv_data.append({"input_ids": tokenizer_output["input_ids"], "pause": pause, "speaker": sex, "label": label})
                conv_data.append({"input_ids": tokenizer_output["input_ids"], "pause": speed, "speaker": sex, "label": label})
            data.append(conv_data)
            
        return data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]   # 会話単位で返す