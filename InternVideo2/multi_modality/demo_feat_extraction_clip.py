import numpy as np
import os
import io
import cv2

import torch

from demo_config import (Config, eval_dict_leaf)
from demo.utils_clip import (retrieve_text, _frame_from_video, setup_internvideo2)

video = cv2.VideoCapture('demo/example1.mp4')
frames = [x for x in _frame_from_video(video)]

text_candidates = ["Woman wears a white top walking down the street.",
                   "Man in baseball cap is riding in a car at night.",
                   "Girl sharing some street view during wlk",
                   "Muslim woman is organizing clothes into bags.",
                   "Chef peels an onion on a cutting board.",
                   "A woman is introducing her family",
                   "Two women are hanging out in a restaurant together.",
                   "View as seen from the airplane window.",
                   "man giving some financial advice  during walk",
                   "Woman gives a monologue next to a potted plant."]

# instead using stage2 config, here we need to use CLIP to get the features with right dimensions (clip, llama)
config = Config.from_file('demo/internvideo2_clip_config.py')
config = eval_dict_leaf(config)

intern_model, tokenizer = setup_internvideo2(config)

text_feats_tensor, vid_feat = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)
print(f'text_feats_tensor: {text_feats_tensor.shape}')
print(f'vid_feat: {vid_feat.shape}')