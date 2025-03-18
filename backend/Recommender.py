import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_itinerary(preferences, tokenizer, model):
    prompt = f"Plan a 5-day trip based on: {preferences}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

bert_model = BertForSequenceClassification.from_pretrained("../models/fine_tuned_model").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_model = GPT2LMHeadModel.from_pretrained("../models/gpt2-itinerary-generator").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

preferences = {"food": 0.6, "experiences": 0.8, "attractions": 0.7}
print(generate_itinerary(preferences, gpt2_tokenizer, gpt2_model))
