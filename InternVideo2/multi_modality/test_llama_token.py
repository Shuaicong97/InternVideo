from transformers import LlamaTokenizer
from models.backbones.internvideo2 import InternVideo2, LLaMA, Tokenizer
from demo_config import (Config, eval_dict_leaf)

tokenizer = LlamaTokenizer.from_pretrained("/home/stud/shuaicong/forkProject/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b",
                                           local_files_only=True,
                                           legacy=False)
tokenizer.pad_token = " "  # allow padding
tokenizer.add_eos_token = True

config = Config.from_file('demo/internvideo2_clip_config.py')

text = "A woman is introducing her family"
tokens = tokenizer.tokenize(text)
print("Token count1:", len(tokens), tokens, type(tokens))
token_ids = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length").input_ids
print("Token count2:", len(token_ids), token_ids)

tokenizer = Tokenizer()
tokens = tokenizer(text, return_tensors="pt", padding="max_length",
                            max_length=32, truncation=True, add_special_tokens=True,
                            add_eos_token=True)

print("Token count3:", len(tokens), tokens)
