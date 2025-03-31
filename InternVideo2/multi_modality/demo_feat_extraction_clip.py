import numpy as np
import os
import io
import cv2

import torch

from demo_config import (Config, eval_dict_leaf)
from demo.utils_clip import (retrieve_text, _frame_from_video, setup_internvideo2)

video = cv2.VideoCapture('demo/example1.mp4')
frames = [x for x in _frame_from_video(video)]

text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon."]

# instead using stage2 config, here we need to use CLIP to get the features with right dimensions (clip, llama)
config = Config.from_file('demo/internvideo2_clip_config.py')
config = eval_dict_leaf(config)

intern_model, tokenizer = setup_internvideo2(config)

texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)
for t, p in zip(texts, probs):
    print(f'text: {t} ~ prob: {p:.4f}')