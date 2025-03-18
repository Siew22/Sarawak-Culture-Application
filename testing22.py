from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "models/gpt2_finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(prompt):
    # 将 prompt 转为张量
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 生成文本
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=400,    # 可根据需求调整
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        no_repeat_ngram_size=3,
        early_stopping=False,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码生成的张量为字符串
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "Generate a travel itinerary for Malaysia:"
    result = generate_text(prompt)
    print("\nGenerated Itinerary:\n", result)
