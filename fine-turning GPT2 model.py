import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import random

# 加载预训练模型和分词器
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置分词器的pad token（GPT2 默认没有）
tokenizer.pad_token = tokenizer.eos_token

# 读取示例数据文件
def load_text_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 假设示例之间使用 "<SEP>" 分隔
    examples = text.split("<SEP>")
    # 去除空行和前后空格
    examples = [ex.strip() for ex in examples if ex.strip()]
    return examples

# 例如，将数据文件转换为 Hugging Face 数据集格式
data_file = "data/itinerary_samples.txt"
examples = load_text_data(data_file)

# 构造 Dataset 对象
data_dict = {"text": examples}
dataset = Dataset.from_dict(data_dict)

# 可选：打乱数据，并划分训练/验证集
dataset = dataset.shuffle(seed=42)
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 数据预处理：使用分词器进行 tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 使用 DataCollator 来动态填充和 mask
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,           # 根据数据量调整epoch数
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    warmup_steps=100,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=True,                  # 如果支持半精度训练
)

# 构造 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# 开始训练
trainer.train()
trainer.save_model("models/gpt2_trained")
tokenizer.save_pretrained("models/gpt2_trained")
