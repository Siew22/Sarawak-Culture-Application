from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 确保使用了正确的 tokenizer
text = "Hello, how are you?"  # 你可以换成自己的测试文本
inputs = tokenizer(text, return_tensors="pt")  # 生成 token 序列

tokens = inputs["input_ids"]
print(f"Token IDs max: {tokens.max()}, min: {tokens.min()}, vocab size: {tokenizer.vocab_size}")
print(inputs)  # 调试：查看 inputs 是否存在
