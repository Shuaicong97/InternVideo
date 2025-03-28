from configs.data import *
from configs.model import *

# ========================= data ==========================
# NOTE The train_file will not be used during the evaluation

num_workers = 12

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 256
batch_size_test = 64
size_t = 224
max_txt_l = 32

# origin_num_frames = 8

use_half_precision = False
use_bf16 = False

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
model = dict(
    model_cls="InternVideo2_CLIP",
    vision_encoder=dict(
        name="internvideo2",
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.3,
        head_drop_path_rate=0.,
        embed_dim=1408,
        num_heads=16,
        mlp_ratio=48/11,
        init_values=0.1,
        qk_normalization=True,
        depth=40,
        use_flash_attn=use_half_precision,
        use_fused_rmsnorm=use_half_precision,
        use_fused_mlp=use_half_precision,
        fused_mlp_heuristic=1,
        drop_cls_token=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
    ),
    text_encoder=dict(
        use_flash_attn=True,
        transformer_width=4096,
        llama_path="/home/stud/shuaicong/forkProject/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b",
        use_lora=True,
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=True,
    open_vision_clip_projector=True,
    freeze_text=True,
    open_text_projection=False,
    open_text_lora=False,
    tokenizer_path="/home/stud/shuaicong/forkProject/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b",
    vision_ckpt_path="/nfs/data3/shuaicong/InternVideo2/demo_weights/InternVideo2-stage2_1b-224p-f4.pt",
    load_vision_ckpt_from_internvideo2_stage2=True,
    text_ckpt_path="/nfs/data3/shuaicong/InternVideo2/demo_weights/internvl_c_13b_224px.pth",
)

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

gradient_checkpointing = True

# ========================= wandb ==========================
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500
auto_resume = True
pretrained_path = "/nfs/data3/shuaicong/InternVideo2/demo_weights/1B_clip.pth"  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=False,
    stage=0,
)