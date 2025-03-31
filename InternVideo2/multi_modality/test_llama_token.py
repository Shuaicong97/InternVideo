from transformers import LlamaTokenizer
from models.backbones.internvideo2 import InternVideo2, LLaMA, Tokenizer
from demo_config import (Config, eval_dict_leaf)

tokenizer = LlamaTokenizer.from_pretrained("/home/stud/shuaicong/forkProject/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b")
config = Config.from_file('demo/internvideo2_clip_config.py')

text = "A woman is introducing her family"
tokens = tokenizer.tokenize(text)
print("Token count1:", len(tokens), tokens)
inputs = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=32,
    return_tensors="pt"
).to(config.device)
print("input_ids shape:", inputs["input_ids"].shape)  # 🔹 torch.Size([1, 32])
print("input_ids[0]:", inputs["input_ids"][0])

tokenizer = Tokenizer()
tokens = tokenizer(text, return_tensors="pt", padding="max_length",
                            max_length=32, truncation=True, add_special_tokens=True,
                            add_eos_token=True)

print("Token count3:", len(tokens), tokens)
