# modify from /InternVideo/InternVideo2/multi_modality/tasks_clip/pretrain.py
import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

import os
import sys
sys.path.append('/home/stud/shuaicong/forkProject/InternVideo/InternVideo2/multi_modality')

from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader_rs, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from tasks_clip.retrieval_utils import evaluation_wrapper
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
# from models.internvideo2_clip import encode_vision

from clip_loader import get_video_loader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np

logger = logging.getLogger(__name__)
world_size = 1  # 例如，你有1个进程
rank = 0  # 当前进程的排名，从0开始
# 设置环境变量
os.environ['RANK'] = str(rank)
os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
os.environ['LOCAL_RANK'] = str(rank)  # 假设每个节点只有一个GPU

def test_transform_init():
    # from /InternVideo/InternVideo2/multi_modality/dataset/__init__.py L133-154
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image   , # NxHxWx3, (16, 480, 640, 3)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform

def main(config):

    # get model
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)


    config.scheduler.num_training_steps = 1 * config.scheduler.epochs
    config.scheduler.num_warmup_steps = 1 * config.scheduler.warmup_epochs

    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=is_pretrain,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    # prepare data
    video_loader = get_video_loader()
    transform=test_transform_init()

    video_dir = "/nfs/data3/shuaicong/data_processing/videos_ovis/valid"
    output_dir = "/nfs/data3/shuaicong/InternVideo2/outputs/ovis_features/ovis_vision_feature_val"
    processed_list_path = os.path.join(output_dir, "processed_files.txt")
    clip_duration = 2  # 每个 clip 长度（单位：秒）
    target_model_frames = 8  # 模型要求8帧输入
    os.makedirs(output_dir, exist_ok=True)

    # 读取已处理文件列表
    if os.path.exists(processed_list_path):
        with open(processed_list_path, "r") as f:
            processed_files = set(line.strip() for line in f.readlines())
    else:
        processed_files = set()

    # 遍历视频文件
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4") and not f.startswith("._")]
    video_files.sort()  # 可选：保证顺序处理

    for filename in video_files:
        video_name = os.path.splitext(filename)[0]
        if video_name in processed_files:
            print(f"跳过已处理视频：{filename}")
            continue

        video_path = os.path.join(video_dir, filename)
        print(f"处理视频：{filename}")

        vr = video_loader(video_path)
        fps = vr.get_avg_fps()  # 例如返回 1.0
        total_frames = len(vr)
        duration_sec = total_frames / fps
        frames_per_clip = int(fps * clip_duration)
        num_clips = int(duration_sec // clip_duration)
        feature_list = []

        for i in range(num_clips):
            start_frame = i * frames_per_clip
            end_frame = start_frame + frames_per_clip
            if end_frame > total_frames:
                break  # 超出帧数了

            frame_indices = np.arange(start_frame, end_frame)  # 提取当前clip的帧
            data = vr.get_batch(frame_indices).numpy()  # NxHxWx3
            data = np.transpose(data, (0, 3, 1, 2))  # (T, 1 or 3, H, W) where T=1 for image
            frame = torch.from_numpy(data)
            frame_q = transform(frame)  # 应该是返回 (T, 3, 224, 224)

            # pad/repeat到 8 帧
            if frame_q.shape[0] < target_model_frames:
                repeats = target_model_frames // frame_q.shape[0]
                pad = target_model_frames - repeats * frame_q.shape[0]
                frame_q = frame_q.repeat(repeats, 1, 1, 1)
                if pad > 0:
                    frame_q = torch.cat([frame_q, frame_q[:pad]], dim=0)

            input_data = frame_q.unsqueeze(0).cuda()  # (1, T, 3, 224, 224)

            # print(input_data.shape)  # torch.Size([1, 8, 3, 224, 224])
            with torch.no_grad():
                feature = model.module.encode_vision(input_data)  # out: [1, 768]
                # print(feature.shape)
                feature_list.append(feature.float().cpu().numpy())

        # 拼接为 [video_length / 2, 768]
        vid_feature = np.concatenate(feature_list, axis=0)
        # print(vid_feature.shape)
        file_path = os.path.join(output_dir, f"{video_name}.pt")
        torch.save(torch.from_numpy(vid_feature), file_path)
        print(f"Saved Feat to {file_path}, shape: {vid_feature.shape}")

        # 写入已处理文件名
        with open(processed_list_path, "a") as f:
            f.write(f"{video_name}\n")

    if is_main_process() and config.wandb.enable:
        run.finish()

    # np.save(url, np.vstack(feature_list))
    # print(f'[{idx} / {num_videos}]: save feature on {url}')
    print("done")




if __name__ == "__main__":

    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
