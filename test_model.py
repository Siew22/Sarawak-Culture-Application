import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from main import fine_tune_gpt2  # 确保 main.py 在同一目录下

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_finetuned_gpt2():
    try:
        # 模型路径
        model_path = r"c:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\models\gpt2_finetuned"  # 使用绝对路径
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # 加载分词器和模型
        logger.info(f"Loading fine-tuned GPT-2 model from: {model_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

        # 添加 pad_token_id（避免警告）
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info("Tokenizer and model loaded successfully")

        # 更详细的提示
        prompt = (
            "Generate a 1-day travel itinerary for Kuching, Malaysia.\n"
            "The itinerary must include four sections (Morning, Noon, Afternoon, Evening), "
            "and each section must contain recommendations following this format:\n"
            "- [category]: [Name], address: [Full Address in Malaysia].\n"
            "Categories must be one of: food, attraction, experience.\n"
            "Include 2 food, 2 attraction, and 1 experience recommendations.\n"
            "All recommendations must be unique and based on real places in Kuching, Malaysia.\n"
            "## Example Format:\n"
            "## Day 1\n"
            "### Morning\n"
            "- attraction: Sarawak Cultural Village, address: Pantai Damai Santubong, 93050 Kuching, Sarawak, Malaysia\n"
            "### Noon\n"
            "- food: Try Sarawak Laksa, address: Top Spot Food Court, Jalan Padungan, 93100 Kuching, Sarawak, Malaysia\n"
            "### Afternoon\n"
            "- experience: Sarawak River Cruise, address: Kuching Waterfront, 93000 Kuching, Sarawak, Malaysia\n"
            "### Evening\n"
            "- attraction: Semenggoh Nature Reserve, address: Jalan Puncak Borneo, 93250 Kuching, Sarawak, Malaysia\n\n"
            "Now generate a 1-day itinerary for Day 1:\n"
        )
        logger.info(f"Using prompt:\n{prompt[:200]}...")

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 生成行程
        logger.info("Generating itinerary...")
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=400,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            early_stopping=False,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
        )

        # 解码生成结果
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info("Itinerary generated successfully")

        # 验证生成的行程
        if "## Day 1" not in generated_text:
            logger.warning("Generated itinerary does not contain '## Day 1', may be incomplete")
        else:
            logger.info("Generated itinerary format looks correct")

        print("\nGenerated Itinerary:\n", generated_text)

    except Exception as e:
        logger.error(f"Error in test_finetuned_gpt2: {str(e)}")
        raise

if __name__ == "__main__":
    # 确保微调和测试都运行
    sample_file = r"c:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\itinerary_samples.txt"  # 使用绝对路径
    skip_finetuning = True  # 设置为 True 以跳过微调，直接测试现有模型

    if not skip_finetuning:
        try:
            logger.info(f"Starting fine-tuning with data file: {sample_file}")
            fine_tune_gpt2(sample_file, epochs=5, batch_size=2)
            logger.info("Fine-tuning completed")
        except Exception as e:
            logger.error(f"Failed to fine-tune GPT-2: {str(e)}")
            raise
    else:
        logger.info("Skipping fine-tuning, testing existing model in: {model_path}")

    # 测试生成
    try:
        logger.info("Starting test of fine-tuned GPT-2 model")
        test_finetuned_gpt2()
    except Exception as e:
        logger.error(f"Failed to test fine-tuned GPT-2: {str(e)}")
        raise