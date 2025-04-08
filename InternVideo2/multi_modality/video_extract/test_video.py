import sys
import os

sys.path.append('/home/stud/shuaicong/forkProject/InternVideo/InternVideo2/multi_modality')

video_path = "/home/stud/shuaicong/forkProject/InternVideo/InternVideo2/multi_modality/demo/example1.mp4"

if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

print(video_path)