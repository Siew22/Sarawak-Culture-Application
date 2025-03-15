import os
import time
import torch
import requests
import pandas as pd
import logging
import re
import traceback
import random
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from datasets import Dataset as HF_Dataset
from fastapi import FastAPI, HTTPException, Query
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple
from fastapi import FastAPI, Query

# ---------------------- 添加 MySQL 连接依赖 ----------------------
import mysql.connector
from mysql.connector import Error

# ---------------------- 环境设置与日志 ----------------------
load_dotenv(dotenv_path=".env")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"CUDA available, using device: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available, falling back to CPU")

# ---------------------- 数据定义 ----------------------
ATTRACTIONS = {
    "Kuching": [
        {"name": "猫博物馆", "address": "Jalan Tun Ahmad Zaidi Adruce, 93400 Kuching, Sarawak, 马来西亚"},
        {"name": "沙捞越文化村", "address": "Pantai Damai, 93752 Kuching, Sarawak, 马来西亚"},
        {"name": "古晋旧法院", "address": "Jalan Tun Abang Haji Openg, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "猫城广场", "address": "Jalan Main Bazaar, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "古晋滨水区", "address": "Kuching Waterfront, 93000 Kuching, Sarawak, 马来西亚"}
    ]
}
FOODS = {
    "Kuching": [
        {"name": "沙捞越叻沙", "address": "Jalan Padungan, 93100 Kuching, Sarawak, 马来西亚"},
        {"name": "马来西亚肉骨茶", "address": "Jalan Song, 93350 Kuching, Sarawak, 马来西亚"},
        {"name": "沙捞越层糕", "address": "Jalan India, 93100 Kuching, Sarawak, 马来西亚"},
        {"name": "三层肉饭", "address": "Main Bazaar, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "古早味面", "address": "Jalan Carpenter, 93000 Kuching, Sarawak, 马来西亚"}
    ]
}
EXPERIENCES = {
    "Kuching": [
        {"name": "拜访伊班族长屋", "address": "Batang Ai, Sarawak, 马来西亚"},
        {"name": "婆罗洲雨林徒步", "address": "Bako National Park, 93050 Kuching, Sarawak, 马来西亚"},
        {"name": "游览砂拉越河", "address": "Kuching Waterfront, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "探索风洞国家公园", "address": "Gunung Mulu National Park, Sarawak, 马来西亚"},
        {"name": "夜市探险", "address": "Jalan Satok, 93400 Kuching, Sarawak, 马来西亚"}
    ]
}

# 定义样本文件的默认路径
DEFAULT_SAMPLE_FILE_PATH = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\itinerary_samples.txt"

# ---------------------- BERT Preference Prediction Module ----------------------
bert_model_path = os.path.join("models", "bert_classifier")
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
if os.path.exists(bert_model_path):
    logger.info(f"Loading BERT model from: {bert_model_path}")
    model_bert = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=2).to(device)
else:
    logger.info("No trained BERT model found, loading pre-trained model for training...")
    model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
model_bert.eval()

def predict_preference(text: str) -> int:
    with torch.no_grad():
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        outputs = model_bert(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

def get_user_preferences(excel_input: str) -> Tuple[Dict, Dict]:
    try:
        logger.debug(f"Attempting to load preferences file: {excel_input}")
        if not os.path.exists(excel_input):
            raise FileNotFoundError(f"File not found: {excel_input}")

        df = pd.read_excel(excel_input, header=None, dtype=str)
        df = df.fillna("")
        logger.debug(f"Excel file content:\n{df.to_string()}")

        preferences_by_day = {}
        extra_preferences = {}
        current_day = None
        current_prefs = []
        in_extra_section = False

        for index, row in df.iterrows():
            if all(val == "" for val in row):
                continue
            if str(row[0]).strip().lower().startswith("day"):
                if current_day is not None and current_prefs:
                    preferences_by_day[current_day] = current_prefs
                day_value = str(row[0]).strip().lower().replace("day", "").strip()
                if day_value and day_value.isdigit():
                    current_day = int(day_value)
                else:
                    raise ValueError(f"Row {index}: Invalid 'Day' value, expected a number, got {day_value}")
                current_prefs = []
                in_extra_section = False
                logger.debug(f"Found Day: {current_day}")
                continue

            if str(row[0]).strip().lower() == "location" and str(row[1]).strip().lower() == "days":
                continue

            if str(row[1]).strip().lower().startswith("experiences"):
                if current_day is not None and current_prefs:
                    preferences_by_day[current_day] = current_prefs
                current_day = None
                current_prefs = []
                in_extra_section = True
                logger.debug("Entering extra preferences section")
                continue

            if row[1] and row[2]:
                category = str(row[1]).strip().lower().replace("s", "")
                try:
                    rate = float(row[2])
                    if rate < 0:
                        rate = 0
                        logger.warning(f"Row {index}: negative rate found, set to 0")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Row {index}: 'rate' must be numeric, got {row[2]}")
                logger.debug(f"Parsing category: {category}, rate: {rate}")
                if in_extra_section:
                    extra_preferences[category] = rate
                elif current_day is not None:
                    current_prefs.append((category, rate))

        if current_day is not None and current_prefs:
            preferences_by_day[current_day] = current_prefs

        logger.debug(f"Extracted preferences by day: {preferences_by_day}")
        logger.debug(f"Extracted extra preferences: {extra_preferences}")
        if not preferences_by_day and not extra_preferences:
            raise ValueError("No valid preferences extracted from the file.")
        logger.info(f"Successfully extracted preferences: {preferences_by_day}, extra: {extra_preferences}")
        return preferences_by_day, extra_preferences

    except Exception as e:
        logger.error(f"Error in get_user_preferences: {str(e)}\n{traceback.format_exc()}")
        raise

def train_bert(excel_path: str, epochs: int = 15, batch_size: int = 4):
    try:
        preferences_by_day, extra_preferences = get_user_preferences(excel_path)
        texts = []
        labels = []
        threshold = 2.5

        for day, prefs in preferences_by_day.items():
            for category, rate in prefs:
                texts.append(f"{category} (Day {day})")
                labels.append(1 if rate > threshold else 0)

        for category, rate in extra_preferences.items():
            texts.append(f"{category} (Extra Preference)")
            labels.append(1 if rate > threshold else 0)

        logger.debug(f"Number of preference data: {len(texts)}")
        logger.debug(f"Texts: {texts}")
        logger.debug(f"Labels: {labels}")
        if len(texts) == 0:
            raise ValueError("No preference data available for training.")
        effective_batch_size = min(batch_size, len(texts))

        class BertDataset(Dataset):
            def _init_(self, texts, labels, tokenizer, max_len=256):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def _len_(self):
                return len(self.texts)

            def _getitem_(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)
                input_ids = encoding["input_ids"].squeeze()
                attention_mask = encoding["attention_mask"].squeeze()
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}

        dataset = BertDataset(texts, labels, tokenizer_bert)
        if len(dataset) < effective_batch_size:
            effective_batch_size = 1
            logger.warning(f"Data size {len(dataset)} is less than batch_size {batch_size}, setting batch_size to 1")

        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        optimizer = AdamW(model_bert.parameters(), lr=2e-5)
        model_bert.train().to(device)

        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            for i, batch in enumerate(dataloader):
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model_bert(**batch)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_bert.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    logger.error(f"Runtime error during training (epoch {epoch+1}, batch {i}): {str(e)}\n{traceback.format_exc()}")
                    torch.cuda.empty_cache()
                    continue
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        os.makedirs(bert_model_path, exist_ok=True)
        model_bert.save_pretrained(bert_model_path)
        tokenizer_bert.save_pretrained(bert_model_path)
        logger.info(f"BERT model saved to {bert_model_path}")

    except Exception as e:
        logger.error(f"Error in train_bert: {str(e)}\n{traceback.format_exc()}")
        raise

# ---------------------- GPT-2 Itinerary Generation Module ----------------------
custom_gpt2_model_path = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\models\gpt2_finetuned"
try:
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(custom_gpt2_model_path)
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
except Exception as e:
    logger.error(f"Failed to load tokenizer from {custom_gpt2_model_path}: {str(e)}")
    raise

try:
    if os.path.exists(custom_gpt2_model_path):
        logger.info(f"Loading fine-tuned GPT-2 model from: {custom_gpt2_model_path}")
        model_gpt2 = GPT2LMHeadModel.from_pretrained(custom_gpt2_model_path).to(device)
    else:
        raise FileNotFoundError(f"Fine-tuned model not found at {custom_gpt2_model_path}")
except Exception as e:
    logger.error(f"Failed to load fine-tuned GPT-2 model: {str(e)}")
    raise

class GPT2Dataset(Dataset):
    def _init_(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        input_ids = encoding["input_ids"].squeeze()
        return input_ids

def fine_tune_gpt2(data_file: str, epochs: int = 5, batch_size: int = 2):
    try:
        logger.debug(f"Attempting to fine-tune GPT-2 with data file: {data_file}")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Sample file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            text = f.read()
        examples = text.split("Generate a 5-day travel itinerary:")
        examples = [("Generate a 5-day travel itinerary:" + ex).strip() for ex in examples if ex.strip()]
        if not examples:
            raise ValueError("No training examples found in the data file.")

        data_dict = {"text": examples}
        hf_dataset = HF_Dataset.from_dict(data_dict)
        hf_dataset = hf_dataset.shuffle(seed=42)
        split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        def tokenize_function(examples):
            return tokenizer_gpt2(examples["text"], truncation=True, max_length=512)

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_gpt2, mlm=False)

        training_args = TrainingArguments(
            output_dir="models/gpt2_finetuned",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            logging_steps=50,
            learning_rate=5e-5,
            warmup_steps=100,
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = Trainer(
            model=model_gpt2,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model("models/gpt2_finetuned")
        tokenizer_gpt2.save_pretrained("models/gpt2_finetuned")
        logger.info("GPT-2 fine-tuning completed and saved to models/gpt2_finetuned")

    except Exception as e:
        logger.error(f"Error in fine_tune_gpt2: {str(e)}\n{traceback.format_exc()}")
        raise

# ---------------------- 辅助函数 ----------------------
def normalize_text(text: str) -> str:
    text = text.replace("Address(s):", "address:")
    text = text.replace("Address:", "address:")
    return text.lower()

def extract_locations(itinerary: str) -> Dict[str, str]:
    logger.debug(f"Extracting locations, input itinerary: {itinerary}")
    locations = {}
    normalized_itinerary = normalize_text(itinerary)
    pattern = re.compile(r"-\s*(food|experience|attraction):\s*([^,]+),\s*address:\s*(.+)", re.IGNORECASE)
    for line in normalized_itinerary.split("\n"):
        match = pattern.search(line.strip())
        if match:
            category, name, address = match.groups()
            locations[name.strip()] = address.strip()
    logger.debug(f"Extracted locations: {locations}")
    return locations

def geocode_location(location: str) -> Tuple[float, float]:
    cache_file = "location_cache.txt"
    cache = {}
    manual_coords = {
        "7 jalan legoland, 79100 nusajaya, johor, malaysia": (1.5028, 103.6314),
        "jalan balik pulau, 11500 air itam, penang, malaysia": (5.4044, 100.2762),
        "pantai damai santubong, 93050 kuching, sarawak, malaysia": (1.7167, 110.3167),
        "siam road char koay teow, 82 jalan siam, 10400 george town, penang, malaysia": (5.4226, 100.3251),
        "gua tempurung, 31600 gopeng, perak, malaysia": (4.4149, 101.1879),
        "top spot food court, jalan padungan, 93100 kuching, sarawak, malaysia": (1.5593, 110.3442),
        "kuching waterfront, 93000 kuching, sarawak, malaysia": (1.5595, 110.3467),
        "pantai damai santubong, 93050 kuching, sarawak, malaysia": (1.7167, 110.3167),
        "jalan puncak borneo, 93250 kuching, sarawak, malaysia": (1.4131, 110.2847),
        "kuala lumpur city centre, 50088 kuala lumpur, malaysia": (3.1579, 101.7123),
        "5, jalan ss 21/37, damansara utama, 47400 petaling jaya, malaysia": (3.1353, 101.6235),
        "gombak, 68100 batu caves, selangor, malaysia": (3.2379, 101.6811),
        "jalan puncak, 50250 kuala lumpur, malaysia": (3.1488, 101.7051),
        "gunung gading national park": (1.69, 109.85),
        "semenggoh wildlife centre": (1.39, 110.31),
        # 新增 Sin Lian Shin 的坐标
        "sin lian shin, jalan sekama, 93300 kuching, sarawak, 马来西亚": (1.5532, 110.3645),
        "sin lian shin": (1.5532, 110.3645),  # 直接名称匹配
    }
    norm_location = location.strip().lower()
    if norm_location == "to be added":
        return None
    if norm_location in manual_coords:
        logger.debug(f"Using manual coordinates for {norm_location}: {manual_coords[norm_location]}")
        return manual_coords[norm_location]
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            for line in f:
                key, lat, lon = line.strip().split("|")
                cache[key] = [float(lat), float(lon)]
    if norm_location in cache:
        logger.debug(f"Using cached coordinates for {norm_location}: {cache[norm_location]}")
        return tuple(cache[norm_location])
   
    # 优化地址查询
    query_location = norm_location
    if "malaysia" not in norm_location:
        query_location = f"{norm_location}, malaysia"
    # 如果是名称，尝试从 FOODS/ATTRACTIONS/EXPERIENCES 中查找完整地址
    full_address = None
    for category_dict in [FOODS, ATTRACTIONS, EXPERIENCES]:
        if location in category_dict:
            for item in category_dict[location]:
                if item["name"].lower() == norm_location:
                    full_address = item["address"].lower()
                    break
    if full_address:
        query_location = full_address
        logger.debug(f"Found full address for {norm_location}: {query_location}")

    params = {"q": query_location, "format": "json", "limit": 1}
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search", params=params,
                               headers={'User-Agent': 'TravelAssistant/2.0'}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            cache[norm_location] = [lat, lon]
            with open(cache_file, "w") as f:
                for k, v in cache.items():
                    f.write(f"{k}|{v[0]}|{v[1]}\n")
            logger.debug(f"Geocoded {norm_location} to ({lat}, {lon})")
            return (lat, lon)
        else:
            logger.warning(f"Geocoding failed for: {query_location}")
            return None
    except Exception as e:
        logger.error(f"Geocoding error for {query_location}: {str(e)}")
        return None

def get_navigation(start: str, end: str) -> Dict:
    api_key = os.getenv("ORS_API_KEY")
    if not api_key:
        logger.warning("ORS_API_KEY not set in .env file")
        return {"error": "Please set ORS_API_KEY in .env file"}

    start_coord = geocode_location(start)
    end_coord = geocode_location(end)
    if not start_coord or not end_coord:
        error_detail = "Geocoding failed for"
        if not start_coord:
            error_detail += f" start: {start}"
        if not end_coord:
            error_detail += f" end: {end}" if not start_coord else f", end: {end}"
        logger.error(error_detail)
        return {"error": error_detail}

    start_lon, start_lat = start_coord[1], start_coord[0]
    end_lon, end_lat = end_coord[1], end_coord[0]

    is_borneo = 109 <= start_lon <= 115 and 0.5 <= start_lat <= 5
    is_peninsula = 100 <= end_lon <= 103 and 1 <= end_lat <= 7
    is_cross_island = (is_borneo and is_peninsula) or (is_peninsula and is_borneo)

    profiles = ["foot-walking", "driving-car"]
    recommendations = {}

    for profile in profiles:
        url = (f"https://api.openrouteservice.org/v2/directions/{profile}?"
               f"api_key={api_key}&start={start_lon},{start_lat}&end={end_lon},{end_lat}&format=json")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Navigation response for {profile}: {data}")

            if "features" in data and data["features"]:
                route = data["features"][0]["properties"]["segments"][0]
                duration_minutes = route["duration"] / 60
                distance_km = route["distance"] / 1000

                if profile == "foot-walking" and (duration_minutes > 1000 or distance_km > 500):
                    recommendations[profile] = {"error": "Walking not feasible for this distance"}
                else:
                    recommendations[profile] = {
                        "duration_minutes": round(duration_minutes, 2),
                        "distance_km": round(distance_km, 2)
                    }
            else:
                recommendations[profile] = {"error": "No route found for this mode"}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                recommendations[profile] = {"error": f"Mode not supported in this region: {str(e)}"}
            else:
                logger.error(f"Navigation request failed for {profile}: {str(e)}")
                recommendations[profile] = {"error": f"Request failed: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Navigation request failed for {profile}: {str(e)}")
            recommendations[profile] = {"error": f"Request failed: {str(e)}"}

    if is_cross_island or (start.lower() == "kuching" and end.lower() == "kuala lumpur"):
        recommendations = {
            "message": "Direct travel between the locations is not possible by walking, driving, or public transport due to geographic separation. Consider taking a flight or other transport."
        }

    return {"recommendations": recommendations}

def format_navigation(nav_data: Dict) -> str:
    if "error" in nav_data:
        return f"Error: {nav_data['error']}"
   
    recommendations = nav_data.get("recommendations", {})
    if "message" in recommendations:
        return recommendations["message"]

    formatted_output = "Navigation Recommendations:\n"
    for profile, details in recommendations.items():
        if "error" in details:
            formatted_output += f"- {profile.replace('-', ' ').title()}: {details['error']}\n"
        else:
            mode = profile.replace('-', ' ').title()
            duration = details["duration_minutes"]
            distance = details["distance_km"]
            formatted_output += f"- {mode}: {duration:.1f} mins, {distance:.2f} km\n"
   
    return formatted_output.strip()

def build_prompt(location: str, day: int, food_count: int, attraction_count: int, experience_count: int, sample_file: str) -> str:
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            samples = [s.strip() for s in content.split("20-DAY TRAVEL ITINERARY:") if s.strip()]
            logger.debug(f"Total samples extracted: {len(samples)}")

            if not samples:
                logger.warning(f"No valid samples found in {sample_file}, using default prompt")
                sample_context = (
                    "DAY 1\n"
                    "MORNING\n"
                    "ATTRACTION: SEMENGGOH WILDLIFE CENTRE, observe orangutans, Address: Semenggoh, 93250 Kuching, Sarawak, Malaysia\n"
                    "FOOD: SARAWAK LAKSA, Address: Choon Hui Cafe, Jalan Ban Hock, 93100 Kuching, Sarawak, Malaysia\n"
                    "NOON\n"
                    "EXPERIENCE: Orangutan feeding session, Address: Semenggoh Wildlife Centre, 93250 Kuching, Sarawak, Malaysia\n"
                    "ATTRACTION: KUCHING WATERFRONT, explore the riverside, Address: 93000 Kuching, Sarawak, Malaysia\n"
                    "AFTERNOON\n"
                    "FOOD: KOLO MEE, Address: Sin Lian Shin, Jalan Sekama, 93300 Kuching, Sarawak, Malaysia\n"
                    "ATTRACTION: SARAWAK STATE MUSEUM, delve into local history, Address: Jalan Tun Abang Haji Openg, 93000 Kuching, Sarawak, Malaysia\n"
                    "EVENING\n"
                    "EXPERIENCE: Sunset cruise on Sarawak River, Address: Kuching Waterfront, 93000 Kuching, Sarawak, Malaysia\n"
                    "FOOD: GRILLED FISH, Address: Top Spot Food Court, Jalan Padungan, 93100 Kuching, Sarawak, Malaysia"
                )
            else:
                relevant_samples = []
                normalized_location = location.split(",")[0].strip().upper()
                for sample in samples:
                    if normalized_location in sample:
                        relevant_samples.append(sample)
                        if len(relevant_samples) >= 2:
                            break
               
                if not relevant_samples:
                    relevant_samples = random.sample(samples, min(2, len(samples)))
               
                # 限制样本长度，避免超过 token 限制
                sample_context = "\n\n".join(relevant_samples[:2])[:1000]  # 限制前 1000 字符
                logger.debug(f"Selected sample context for {location}:\n{sample_context}")
    except Exception as e:
        logger.error(f"Error reading sample file {sample_file}: {str(e)}")
        sample_context = (
            "DAY 1\n"
            "MORNING\n"
            "ATTRACTION: SEMENGGOH WILDLIFE CENTRE, observe orangutans, Address: Semenggoh, 93250 Kuching, Sarawak, Malaysia\n"
            "FOOD: SARAWAK LAKSA, Address: Choon Hui Cafe, Jalan Ban Hock, 93100 Kuching, Sarawak, Malaysia\n"
            "NOON\n"
            "EXPERIENCE: Orangutan feeding session, Address: Semenggoh Wildlife Centre, 93250 Kuching, Sarawak, Malaysia\n"
            "ATTRACTION: KUCHING WATERFRONT, explore the riverside, Address: 93000 Kuching, Sarawak, Malaysia\n"
            "AFTERNOON\n"
            "FOOD: KOLO MEE, Address: Sin Lian Shin, Jalan Sekama, 93300 Kuching, Sarawak, Malaysia\n"
            "ATTRACTION: SARAWAK STATE MUSEUM, delve into local history, Address: Jalan Tun Abang Haji Openg, 93000 Kuching, Sarawak, Malaysia\n"
            "EVENING\n"
            "EXPERIENCE: Sunset cruise on Sarawak River, Address: Kuching Waterfront, 93000 Kuching, Sarawak, Malaysia\n"
            "FOOD: GRILLED FISH, Address: Top Spot Food Court, Jalan Padungan, 93100 Kuching, Sarawak, Malaysia"
        )

    prompt = (
        f"Generate a 1-day travel itinerary for {location.upper()} with the following requirements:\n"
        f"- FOOD: {food_count} recommendations\n"
        f"- ATTRACTION: {attraction_count} recommendations\n"
        f"- EXPERIENCE: {experience_count} recommendations\n"
        "Format:\n"
        "DAY X\n"
        "MORNING\n"
        "ATTRACTION: [Name], [description], Address: [Full Address in Malaysia]\n"
        "FOOD: [Name], Address: [Full Address in Malaysia]\n"
        "NOON\n"
        "EXPERIENCE: [Name], Address: [Full Address in Malaysia]\n"
        "ATTRACTION: [Name], [description], Address: [Full Address in Malaysia]\n"
        "AFTERNOON\n"
        "FOOD: [Name], Address: [Full Address in Malaysia]\n"
        "ATTRACTION: [Name], [description], Address: [Full Address in Malaysia]\n"
        "EVENING\n"
        "EXPERIENCE: [Name], Address: [Full Address in Malaysia]\n"
        "FOOD: [Name], Address: [Full Address in Malaysia]\n"
        "Rules:\n"
        "1. Distribute activities evenly across MORNING, NOON, AFTERNOON, and EVENING.\n"
        "2. Use complete and realistic addresses (e.g., 'Jalan Sultan, 50000 Kuala Lumpur, Malaysia').\n"
        "3. Do NOT repeat locations or use placeholders.\n"
        "4. Use English for all text, and capitalize section names (e.g., MORNING, FOOD).\n\n"
        "Examples:\n"
        f"{sample_context}\n\n"
        f"Now generate a 1-day itinerary for DAY {day} in {location.upper()}:\n"
    )
    logger.debug(f"Prompt for Day {day} (first 2000 chars):\n{prompt[:2000]}...")
    return prompt

def distribute_recommendations(available_foods, available_attractions, available_experiences, days, food_value, attraction_value, experience_value):
    # 确保返回的是列表
    daily_foods = [food_value] * days
    daily_attractions = [attraction_value] * days
    daily_experiences = [experience_value] * days
    return daily_foods, daily_attractions, daily_experiences

def enforce_exact_recommendations(
    day_text: str,
    food_count: int,
    attraction_count: int,
    experience_count: int,
    day: int,
    location: str,
    all_days_counts: Tuple[List, List, List],  # 确保是列表
    day_index: int,
    available_attractions: List[Dict] = None,
    available_foods: List[Dict] = None,
    available_experiences: List[Dict] = None
) -> str:
    available_attractions = available_attractions or []
    available_foods = available_foods or []
    available_experiences = available_experiences or []
    
    # 确保活动数量足够
    if len(available_attractions) < attraction_count:
        logger.warning(f"Insufficient attractions for DAY {day}: {len(available_attractions)} available, {attraction_count} needed. Repeating activities.")
        available_attractions.extend(random.sample(available_attractions, attraction_count - len(available_attractions)) if available_attractions else [{"name": f"Placeholder Attraction {i}", "address": f"Unknown Address, {location}"} for i in range(attraction_count - len(available_attractions))])
    if len(available_foods) < food_count:
        logger.warning(f"Insufficient foods for DAY {day}: {len(available_foods)} available, {food_count} needed. Repeating activities.")
        available_foods.extend(random.sample(available_foods, food_count - len(available_foods)) if available_foods else [{"name": f"Placeholder Food {i}", "address": f"Unknown Address, {location}"} for i in range(food_count - len(available_foods))])
    if len(available_experiences) < experience_count:
        logger.warning(f"Insufficient experiences for DAY {day}: {len(available_experiences)} available, {experience_count} needed. Repeating activities.")
        available_experiences.extend(random.sample(available_experiences, experience_count - len(available_experiences)) if available_experiences else [{"name": f"Placeholder Experience {i}", "address": f"Unknown Address, {location}"} for i in range(experience_count - len(available_experiences))])
    
    logger.debug(f"Available attractions: {len(available_attractions)}")
    logger.debug(f"Available foods: {len(available_foods)}")
    logger.debug(f"Available experiences: {len(available_experiences)}")
    
    daily_foods, daily_attractions, daily_experiences = all_days_counts
    this_day_food = daily_foods[day_index]
    this_day_attraction = daily_attractions[day_index]
    this_day_experience = daily_experiences[day_index]
    logger.debug(f"Counts for DAY {day}: food={this_day_food}, attraction={this_day_attraction}, experience={this_day_experience}")
    
    # 选择活动（避免重复）
    selected_attractions = random.sample(available_attractions, this_day_attraction)
    selected_foods = random.sample(available_foods, this_day_food)
    selected_experiences = random.sample(available_experiences, this_day_experience)
    
    logger.debug(f"Selected attractions: {selected_attractions}")
    logger.debug(f"Selected foods: {selected_foods}")
    logger.debug(f"Selected experiences: {selected_experiences}")
    
    # 按类型分组活动
    attraction_activities = [f"ATTRACTION: {item['name']}, Address: {item['address']}" for item in selected_attractions]
    food_activities = [f"FOOD: {item['name']}, Address: {item['address']}" for item in selected_foods]
    experience_activities = [f"EXPERIENCE: {item['name']}, Address: {item['address']}" for item in selected_experiences]
    
    # 分配活动到时段，确保每种类型都满足需求
    time_slots = ["MORNING", "NOON", "AFTERNOON", "EVENING"]
    slot_allocations = {slot: [] for slot in time_slots}
    total_activities = this_day_food + this_day_attraction + this_day_experience
    activities_per_slot = max(3, total_activities // len(time_slots))  # 至少 3 个，最多 5 个
    
    # 优先分配，确保每种类型均匀分布
    remaining_foods = food_activities.copy()
    remaining_attractions = attraction_activities.copy()
    remaining_experiences = experience_activities.copy()
    
    for slot in time_slots:
        # 分配美食
        for _ in range(min(2, this_day_food // len(time_slots) + (1 if len(remaining_foods) > 0 else 0))):
            if remaining_foods:
                slot_allocations[slot].append(remaining_foods.pop(0))
        # 分配景点
        for _ in range(min(2, this_day_attraction // len(time_slots) + (1 if len(remaining_attractions) > 0 else 0))):
            if remaining_attractions:
                slot_allocations[slot].append(remaining_attractions.pop(0))
        # 分配体验
        for _ in range(min(2, this_day_experience // len(time_slots) + (1 if len(remaining_experiences) > 0 else 0))):
            if remaining_experiences:
                slot_allocations[slot].append(remaining_experiences.pop(0))
    
    # 分配剩余活动
    remaining_activities = remaining_foods + remaining_attractions + remaining_experiences
    random.shuffle(remaining_activities)
    for slot in time_slots:
        while remaining_activities and len(slot_allocations[slot]) < 5:  # 允许每时段最多 5 个活动
            slot_allocations[slot].append(remaining_activities.pop(0))
    
    # 确保所有活动都被分配
    while remaining_activities:
        for slot in time_slots:
            if remaining_activities:
                slot_allocations[slot].append(remaining_activities.pop(0))
    
    # 验证分配数量
    total_allocated = sum(len(activities) for activities in slot_allocations.values())
    if total_allocated < total_activities:
        logger.warning(f"Insufficient activities allocated for DAY {day}: {total_allocated} allocated, {total_activities} needed. Adding placeholders.")
        while total_allocated < total_activities:
            for slot in time_slots:
                if len(slot_allocations[slot]) < 5:  # 允许更多活动
                    slot_allocations[slot].append(f"FOOD: Placeholder, Address: {location}")
                    total_allocated += 1
                    if total_allocated >= total_activities:
                        break
    
    new_day = f"DAY {day}\n"
    for slot, slot_activities in slot_allocations.items():
        new_day += f"{slot}\n"
        if slot_activities:
            for activity in slot_activities:
                new_day += f"{activity}\n"
        else:
            new_day += "to be added\n"
    
    logger.debug(f"Generated DAY {day} itinerary:\n{new_day}")
    return new_day

def correct_itinerary_with_bert(day_text: str) -> str:
    logger.debug("Starting BERT correction...")
    try:
        # 确保模型已加载
        if 'bert_tokenizer' not in globals() or 'bert_model' not in globals():
            logger.debug("Loading BERT tokenizer and model...")
            global bert_tokenizer, bert_model
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            bert_model.eval()
            if torch.cuda.is_available():
                bert_model.cuda()
        # 简化推理逻辑
        return day_text  # 暂时直接返回，避免复杂推理
    except Exception as e:
        logger.error(f"Error in BERT correction: {str(e)}")
        return day_text

def validate_itinerary(itinerary: str, days: int = 1) -> bool:
    # 支持 DAY X 格式
    required_days = [f"DAY {i}" for i in range(1, days + 1)]
    missing_days = [day for day in required_days if day not in itinerary]
    if missing_days:
        logger.warning(f"Missing days: {missing_days}")
        return False
    logger.debug(f"Itinerary validation passed for {days} days")
    return True

def format_itinerary(raw_text: str, days: int = 5) -> str:
    logger.debug(f"Raw itinerary text: {raw_text}")
    try:
        content = raw_text.strip()
        day_sections = re.split(r"##\s*Day\s*\d+", content)[1:]
        if len(day_sections) < days:
            logger.warning(f"Detected {len(day_sections)} days, less than {days}, forcing supplement")
            while len(day_sections) < days:
                day_sections.append(
                    "\n### Morning\n- to be added\n"
                    "### Noon\n- to be added\n"
                    "### Afternoon\n- to be added\n"
                    "### Evening\n- to be added"
                )
        md_output = "## Personalized Travel Itinerary\n\n"
        for i, day_text in enumerate(day_sections[:days], start=1):
            day_text = day_text.strip()
            md_output += f"## Day {i}\n{day_text}\n\n"
        logger.debug(f"Formatted itinerary:\n{md_output}")
        return md_output
    except Exception as e:
        logger.error(f"Itinerary formatting failed: {str(e)}\n{traceback.format_exc()}")
        return raw_text

def generate_itinerary(location: str, days: int, food_value: int, attraction_value: int, experience_value: int, sample_file: str, use_gpt2: bool = False) -> Dict:
    try:
        logger.info(f"Generating itinerary for location: {location}, days: {days}, food_value: {food_value}, attraction_value: {attraction_value}, experience_value: {experience_value}")
        
        # 一次性提取所有活动数据
        available_attractions = []
        available_foods = []
        available_experiences = []
        seen_foods = set()
        seen_attractions = set()
        seen_experiences = set()
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            logger.debug(f"Raw content from {sample_file}: {content[:500]}...")
            samples = [s.strip() for s in re.split(r"20-DAY TRAVEL ITINERARY:", content, flags=re.IGNORECASE) if s.strip()]
            logger.debug(f"Extracted samples: {len(samples)}")
            
            if not samples:
                logger.warning("No samples found in file. Using default activities.")
            else:
                normalized_location = location.split(",")[0].strip().upper()
                logger.debug(f"Normalized location: {normalized_location}")
                matched_samples = []
                for sample in samples:
                    if normalized_location in sample.upper():
                        matched_samples.append(sample)
                
                if not matched_samples:
                    logger.warning(f"No samples matched for location {normalized_location}. Using all available samples.")
                    matched_samples = samples
                
                for sample in matched_samples:
                    logger.debug(f"Processing sample: {sample[:200]}...")
                    for line in sample.split("\n"):
                        line = line.strip()
                        if re.match(r"ATTRACTION:", line, re.IGNORECASE):
                            parts = re.split(r", Address:", line, maxsplit=1)
                            if len(parts) >= 2 and parts[1].strip():
                                name_desc = parts[0].replace("ATTRACTION:", "").strip()
                                address = parts[1].strip()
                                if normalized_location not in address.upper():
                                    logger.debug(f"Skipping attraction due to location mismatch: {name_desc}, Address: {address}")
                                    continue
                                if name_desc not in seen_attractions:
                                    seen_attractions.add(name_desc)
                                    available_attractions.append({"name": name_desc, "address": address})
                            else:
                                logger.warning(f"Invalid attraction format or missing address: {line}")
                        elif re.match(r"FOOD:", line, re.IGNORECASE):
                            parts = re.split(r", Address:", line, maxsplit=1)
                            if len(parts) >= 2 and parts[1].strip():
                                name = parts[0].replace("FOOD:", "").strip()
                                address = parts[1].strip()
                                if normalized_location not in address.upper():
                                    logger.debug(f"Skipping food due to location mismatch: {name}, Address: {address}")
                                    continue
                                if name not in seen_foods:
                                    seen_foods.add(name)
                                    available_foods.append({"name": name, "address": address})
                            else:
                                logger.warning(f"Invalid food format or missing address: {line}")
                        elif re.match(r"EXPERIENCE:", line, re.IGNORECASE):
                            parts = re.split(r", Address:", line, maxsplit=1)
                            if len(parts) >= 2 and parts[1].strip():
                                name = parts[0].replace("EXPERIENCE:", "").strip()
                                address = parts[1].strip()
                                if normalized_location not in address.upper():
                                    logger.debug(f"Skipping experience due to location mismatch: {name}, Address: {address}")
                                    continue
                                if name not in seen_experiences:
                                    seen_experiences.add(name)
                                    available_experiences.append({"name": name, "address": address})
                            else:
                                logger.warning(f"Invalid experience format or missing address: {line}")
        
        logger.debug(f"Extracted activities: attractions={len(available_attractions)}, foods={len(available_foods)}, experiences={len(available_experiences)}")
        
        # 确保活动数量足够
        total_food_needed = food_value * days
        total_attractions_needed = attraction_value * days
        total_experiences_needed = experience_value * days
        
        if len(available_foods) < total_food_needed:
            logger.warning(f"Insufficient foods: {len(available_foods)} available, {total_food_needed} needed. Repeating activities.")
            available_foods.extend(random.sample(available_foods, total_food_needed - len(available_foods)) if available_foods else [{"name": f"Placeholder Food {i}", "address": f"Unknown Address, {location}"} for i in range(total_food_needed - len(available_foods))])
        if len(available_attractions) < total_attractions_needed:
            logger.warning(f"Insufficient attractions: {len(available_attractions)} available, {total_attractions_needed} needed. Repeating activities.")
            available_attractions.extend(random.sample(available_attractions, total_attractions_needed - len(available_attractions)) if available_attractions else [{"name": f"Placeholder Attraction {i}", "address": f"Unknown Address, {location}"} for i in range(total_attractions_needed - len(available_attractions))])
        if len(available_experiences) < total_experiences_needed:
            logger.warning(f"Insufficient experiences: {len(available_experiences)} available, {total_experiences_needed} needed. Repeating activities.")
            available_experiences.extend(random.sample(available_experiences, total_experiences_needed - len(available_experiences)) if available_experiences else [{"name": f"Placeholder Experience {i}", "address": f"Unknown Address, {location}"} for i in range(total_experiences_needed - len(available_experiences))])        
        while len(available_foods) < total_food_needed:
            logger.warning(f"Insufficient foods: {len(available_foods)} available, {total_food_needed} needed. Repeating activities.")
            available_foods.extend(available_foods[:total_food_needed - len(available_foods)])
        while len(available_attractions) < total_attractions_needed:
            logger.warning(f"Insufficient attractions: {len(available_attractions)} available, {total_attractions_needed} needed. Repeating activities.")
            available_attractions.extend(available_attractions[:total_attractions_needed - len(available_attractions)])
        while len(available_experiences) < total_experiences_needed:
            logger.warning(f"Insufficient experiences: {len(available_experiences)} available, {total_experiences_needed} needed. Repeating activities.")
            available_experiences.extend(available_experiences[:total_experiences_needed - len(available_experiences)])
        
        # 调用 distribute_recommendations
        daily_foods, daily_attractions, daily_experiences = distribute_recommendations(
                 available_foods, available_attractions, available_experiences, days, food_value, attraction_value, experience_value
        )
        logger.debug(f"Daily counts: foods={daily_foods}, attractions={daily_attractions}, experiences={daily_experiences}")
        
        itinerary_list = []
        for day in range(1, days + 1):
            day_index = day - 1
            day_text = ""
            
            all_days_counts = (daily_foods, daily_attractions, daily_experiences)
            day_text = enforce_exact_recommendations(
                day_text=day_text,
                food_count=daily_foods[day_index],
                attraction_count=daily_attractions[day_index],
                experience_count=daily_experiences[day_index],
                day=day,
                location=location,
                all_days_counts=all_days_counts,
                day_index=day_index,
                available_attractions=available_attractions,
                available_foods=available_foods,
                available_experiences=available_experiences
            )
            
            if use_gpt2:
                logger.debug("Skipping GPT-2 correction to improve performance")
            
            day_schedule = {"MORNING": {}, "NOON": {}, "AFTERNOON": {}, "EVENING": {}}
            current_slot = None
            for line in day_text.split("\n"):
                line = line.strip()
                if line in ["MORNING", "NOON", "AFTERNOON", "EVENING"]:
                    current_slot = line
                elif line.startswith("FOOD:") and current_slot:
                    if "food" not in day_schedule[current_slot]:
                        day_schedule[current_slot]["food"] = []
                    day_schedule[current_slot]["food"].append(line)
                elif line.startswith("ATTRACTION:") and current_slot:
                    if "attraction" not in day_schedule[current_slot]:
                        day_schedule[current_slot]["attraction"] = []
                    day_schedule[current_slot]["attraction"].append(line)
                elif line.startswith("EXPERIENCE:") and current_slot:
                    if "experience" not in day_schedule[current_slot]:
                        day_schedule[current_slot]["experience"] = []
                    day_schedule[current_slot]["experience"].append(line)
            
            itinerary_list.append({"day": day, "schedule": day_schedule})
        
        # ---------------------- 添加：保存行程到数据库 ----------------------
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                for day_entry in itinerary_list:
                    day = day_entry["day"]
                    for slot, activities in day_entry["schedule"].items():
                        for activity_type, items in activities.items():
                            for item in items:
                                parts = item.split(", Address:", 1)
                                if len(parts) < 2:
                                    logger.warning(f"无效的活动格式: {item}")
                                    continue
                                name = parts[0].split(": ", 1)[1].strip()
                                address = parts[1].strip()
                                type_upper = activity_type.upper()
                                query = """
                                    INSERT INTO itineraries (location, day, slot, type, name, address, generated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """
                                values = (location.upper(), day, slot, type_upper, name, address, datetime.now())
                                cursor.execute(query, values)
                connection.commit()
                logger.info("行程已保存到数据库")
            except Error as e:
                logger.error(f"保存行程到数据库失败: {e}")
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()

        return {
            "itinerary": itinerary_list,
            "location": location.upper(),
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in generate_itinerary: {str(e)}\n{traceback.format_exc()}")
        raise

# Run the itinerary generation
if __name__ == "_main_":
    location = "Kuching, Sarawak"
    days = 3
    food_value = 3
    attraction_value = 3
    experience_value = 2
    
    itinerary = generate_itinerary(location, days, food_value, attraction_value, experience_value)
    print("Generated Itinerary:")
    import json
    print(json.dumps(itinerary, indent=2, ensure_ascii=False))

# ---------------------- FastAPI Application ----------------------
app = FastAPI()

@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {"message": "Welcome to the AI Travel Assistant API"}

import asyncio

app = FastAPI()

@app.get("/generate_itinerary")
async def generate_itinerary_endpoint(
    location: str,
    days: int,
    food_value: int,
    attraction_value: int,
    experience_value: int,
    sample_file: str,
    use_gpt2: bool = False
):
    logger.debug(f"generate_itinerary_endpoint called with location={location}, days={days}, food_value={food_value}, attraction_value={attraction_value}, experience_value={experience_value}, sample_file={sample_file}, use_gpt2={use_gpt2}")
    try:
        # 清理 sample_file 参数，去掉引号
        sample_file = sample_file.strip('"').strip("'")
        logger.debug(f"Cleaned sample_file path: {sample_file}")
        
        # 设置 30 秒超时
        result = await asyncio.wait_for(
            asyncio.to_thread(
                generate_itinerary,
                location,
                days,
                food_value,
                attraction_value,
                experience_value,
                sample_file,
                use_gpt2
            ),
            timeout=30.0
        )
        logger.debug(f"generate_itinerary_endpoint success: {result}")
        return result
    except asyncio.TimeoutError:
        logger.error("Itinerary generation timed out after 30 seconds")
        raise HTTPException(status_code=504, detail="Itinerary generation timed out")
    except Exception as e:
        logger.error(f"Error in generate_itinerary_endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_itinerary_json")
async def generate_itinerary_json_endpoint(
    location: str = Query("Kuching", description="Travel destination"),
    days: int = Query(1, description="Number of days", ge=1),
    food_value: int = Query(2, description="Number of food recommendations (max 5)", ge=0, le=5),
    attraction_value: int = Query(2, description="Number of attraction recommendations (max 5)", ge=0, le=5),
    experience_value: int = Query(1, description="Number of experience recommendations (max 5)", ge=0, le=5),
    sample_file: str = Query(default=DEFAULT_SAMPLE_FILE_PATH, description="Path to sample itinerary file")
):
    logger.debug(f"generate_itinerary_json_endpoint called with location={location}, days={days}, food_value={food_value}, attraction_value={attraction_value}, experience_value={experience_value}, sample_file={sample_file}")
    try:
        logger.debug(f"Checking if sample file exists: {sample_file}")
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"Sample file '{sample_file}' not found")

        result = generate_itinerary(
            location=location,
            days=days,
            food_value=food_value,
            attraction_value=attraction_value,
            experience_value=experience_value,
            sample_file=sample_file
        )
        logger.debug(f"generate_itinerary_json_endpoint success: {result}")
        return result
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in generate_itinerary_json: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating JSON itinerary: {str(e)}")

@app.get("/navigate")
async def navigate_endpoint(
    start: str = Query(..., description="Starting location"),
    end: str = Query(..., description="Ending location"),
):
    logger.debug(f"navigate_endpoint called with start={start}, end={end}")
    try:
        logger.debug(f"Navigating from {start} to {end}")
        nav_data = get_navigation(start, end)
        logger.debug(f"navigate_endpoint success: {nav_data}")
        return nav_data
    except Exception as e:
        logger.error(f"Error in navigate_endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating navigation: {str(e)}")

# ---------------------- 添加：数据库连接函数 ----------------------
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            port=3306,
            ssl_ca=r"DigiCertGlobalRootG2.crt",  # 替换为你的 SSL 证书路径
            ssl_verify_cert=True
        )
        return connection
    except Error as e:
        logger.error(f"数据库连接失败: {e}")
        return None

# ---------------------- 添加：初始化数据库表 ----------------------
def init_db():
    connection = get_db_connection()
    if not connection:
        logger.error("无法初始化数据库")
        return
    
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS itineraries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                location VARCHAR(255) NOT NULL,
                day INT NOT NULL,
                slot VARCHAR(50) NOT NULL,
                type VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL,
                address TEXT NOT NULL,
                generated_at DATETIME NOT NULL
            )
        """)
        connection.commit()
        logger.info("数据库表已初始化")
    except Error as e:
        logger.error(f"初始化数据库失败: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# ---------------------- 添加：从数据库读取行程的端点 ----------------------
@app.get("/get_itineraries")
async def get_itineraries(location: str = Query(..., description="Travel destination")):
    logger.debug(f"get_itineraries called with location={location}")
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="无法连接到数据库")
    
    try:
        cursor = connection.cursor()
        query = "SELECT day, slot, type, name, address FROM itineraries WHERE location = %s ORDER BY day, slot"
        cursor.execute(query, (location.upper(),))
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"No itinerary data found for location: {location}")
            raise HTTPException(status_code=404, detail=f"没有找到 {location} 的行程数据")
        
        itinerary_list = []
        current_day = None
        day_schedule = None
        
        for row in rows:
            day, slot, type_lower, name, address = row
            if current_day != day:
                if day_schedule is not None:
                    itinerary_list.append({"day": current_day, "schedule": day_schedule})
                day_schedule = {"MORNING": {}, "NOON": {}, "AFTERNOON": {}, "EVENING": {}}
                current_day = day
            
            activity = f"{type_lower.upper()}: {name}, Address: {address}"
            type_lower = type_lower.lower()
            if type_lower not in day_schedule[slot]:
                day_schedule[slot][type_lower] = []
            day_schedule[slot][type_lower].append(activity)
        
        if day_schedule is not None:
            itinerary_list.append({"day": current_day, "schedule": day_schedule})
        
        logger.debug(f"get_itineraries success: {itinerary_list}")
        return {
            "itinerary": itinerary_list,
            "location": location.upper(),
            "retrieved_at": datetime.now().isoformat()
        }
    
    except Error as e:
        logger.error(f"查询数据库失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询数据库失败: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# ---------------------- 主程序入口 ----------------------
if __name__ == "_main_":
    sample_file = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\sample_itineraries.txt"
    preference_file = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\user_preferences.xlsx"

    try:
        if os.path.exists(preference_file):
            train_bert(preference_file, epochs=15, batch_size=4)
        else:
            logger.warning(f"Preference file not found, skipping BERT training: {preference_file}")
    except Exception as e:
        logger.error(f"Failed to train BERT: {str(e)}\n{traceback.format_exc()}")

    try:
        if os.path.exists(sample_file):
            fine_tune_gpt2(sample_file, epochs=5, batch_size=2)
        else:
            logger.warning(f"Sample file not found, skipping GPT-2 fine-tuning: {sample_file}")
    except Exception as e:
        logger.error(f"Failed to fine-tune GPT-2: {str(e)}\n{traceback.format_exc()}")

    # ---------------------- 添加：初始化数据库 ----------------------
    init_db()

    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8800)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}\n{traceback.format_exc()}")
        raise

    from fastapi import FastAPI, Depends
    from database import get_db
    from sqlalchemy.orm import Session

    app = FastAPI()

    @app.get("/users")
    def get_users(db: Session = Depends(get_db)):
        result = db.execute("SELECT * FROM users")
        users = result.fetchall()
        return {"users": users}
    
    @app.post("/add_user")
    def add_user(name: str, email: str, db: Session = Depends(get_db)):
        db.execute("INSERT INTO users (name, email) VALUES (:name, :email)", {"name": name, "email": email})
        db.commit()
        return {"message": "User added successfully"}