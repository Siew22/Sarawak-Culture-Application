from transformers import GPT2Tokenizer

# 使用 EleutherAI/gpt-neo-125M 的 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# 定义一个示例文本
text = "这是一个简单的测试文本，用于检查 tokenizer 输出的 tensor 尺寸。"

# 编码文本，设置最大长度为 2048（如果文本不足则不影响结果）
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=2048)

# 现在 inputs 已经被定义，可以获取 input_ids
tokens = inputs["input_ids"]
print("Token shape:", tokens.shape)
