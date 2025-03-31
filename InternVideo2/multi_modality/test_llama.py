from models.backbones.internvideo2.internvideo2_clip_text import Tokenizer

tokenizer = Tokenizer()
text_sample = ["你好，这是一个测试。"]
output = tokenizer(text_sample)
print(output)
