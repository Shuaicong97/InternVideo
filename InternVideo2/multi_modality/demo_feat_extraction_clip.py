import numpy as np
import os
import io
import cv2
import json

import torch

from demo_config import (Config, eval_dict_leaf)
from demo.utils_clip import (retrieve_text, _frame_from_video, setup_internvideo2)

qid_query_pairs = {}

with open("/home/stud/shuaicong/forkProjects/FlashVTG/data/ovis_val_release.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        qid = data.get("qid")
        query = data.get("query")

        if qid is not None and query is not None:
            qid_query_pairs[qid] = query

query_list = list(qid_query_pairs.values())

# print(len(query_list), query_list)

video = cv2.VideoCapture('demo/example1.mp4')
frames = [x for x in _frame_from_video(video)]

text_candidates = ["A woman is introducing her family", "The rabbit leaps over another rabbit",
                   "A dog chases a ball", "A cat sits on the sofa"]

# instead using stage2 config, here we need to use CLIP to get the features with right dimensions (clip, llama)
config = Config.from_file('demo/internvideo2_clip_config.py')
config = eval_dict_leaf(config)

intern_model, tokenizer = setup_internvideo2(config)

packed_feats = retrieve_text(frames, query_list, model=intern_model, config=config)
split_feats = torch.split(packed_feats["tensor"], packed_feats["lengths"], dim=0)
for i, feat in enumerate(split_feats):
    feat_cpu = feat.cpu()
    file_path = os.path.join('/nfs/data3/shuaicong/InternVideo2/outputs/ovis_features/ovis_llama_text_feature_val', f"qid{i + 5095}.pt")
    torch.save(feat_cpu, file_path)
    print(f"Saved Feat {i} to {file_path}, shape: {feat_cpu.shape}")

# print(f'vid_feat: {vid_feat.shape}')