import torch

features = torch.load('0b02886a.pt', weights_only=True)
print(f"Tensor shape: {features.shape}")


# 检查设备
print(features.device)  # 输出: cpu 或 cuda:0