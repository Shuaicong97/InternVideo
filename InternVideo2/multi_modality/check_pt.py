import os
import torch

folder_path = '/root/projects/InternVideo/InternVideo2/multi_modality/video_extract/ovis_vision_feature_final'
folder_path = '/root/projects/InternVideo/InternVideo2/multi_modality/video_extract/mot17_vision_feature_final'
folder_path = '/root/projects/InternVideo/InternVideo2/multi_modality/video_extract/mot20_vision_feature_final'

gpu_files = []

for filename in os.listdir(folder_path):
    if filename.endswith('.pt'):
        path = os.path.join(folder_path, filename)
        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            tensors = data.values() if isinstance(data, dict) else [data]
            if any(isinstance(t, torch.Tensor) and t.device.type == 'cuda' for t in tensors):
                gpu_files.append(filename)
        except Exception as e:
            print(f"{filename}: 加载出错 ({e})")

if gpu_files:
    for f in gpu_files:
        print(f"{f}: 保存在gpu上")
else:
    print("都保存在cpu上")
