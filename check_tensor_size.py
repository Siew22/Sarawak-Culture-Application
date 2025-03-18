from transformers import GPT2Tokenizer

# 加载 tokenizer（这里以 EleutherAI/gpt-neo-125M 为例）
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# 定义一个示例文本
text = "这是一个测试文本，用于检查 tokenizer 输出的 tensor 尺寸。"

# 使用 tokenizer 对文本进行编码，生成输入张量
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=2048)

# 获取 input_ids 并打印形状
tokens = inputs["input_ids"]
print("Token shape:", tokens.shape)
