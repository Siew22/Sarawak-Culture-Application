from fastapi import FastAPI, HTTPException, Query, File, UploadFile
import os
from datetime import datetime
import logging
import json
import random

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 模拟数据库
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

# 修改生成行程的函数
def generate_itinerary(location, days, food_count, attraction_count, experience_count):
    try:
        # 验证地点是否支持
        if location not in ATTRACTIONS or location not in FOODS or location not in EXPERIENCES:
            raise ValueError(f"Location '{location}' is not supported. Supported locations: {list(ATTRACTIONS.keys())}")

        # 获取可用的景点、食物和体验
        available_attractions = ATTRACTIONS.get(location, [])
        available_foods = FOODS.get(location, [])
        available_experiences = EXPERIENCES.get(location, [])

        # 强制补全活动数量
        selected_attractions = random.sample(available_attractions, min(attraction_count, len(available_attractions))) if available_attractions else []
        selected_foods = random.sample(available_foods, min(food_count, len(available_foods))) if available_foods else []
        selected_experiences = random.sample(available_experiences, min(experience_count, len(available_experiences))) if available_experiences else []

        # 如果活动不足，填充占位符
        while len(selected_attractions) < attraction_count:
            selected_attractions.append({"name": "[Placeholder Attraction]", "address": "[Placeholder Address]"})
        while len(selected_foods) < food_count:
            selected_foods.append({"name": "[Placeholder Food]", "address": "[Placeholder Address]"})
        while len(selected_experiences) < experience_count:
            selected_experiences.append({"name": "[Placeholder Experience]", "address": "[Placeholder Address]"})

        logger.debug(f"Selected for {location}: {len(selected_attractions)} attractions, {len(selected_foods)} foods, {len(selected_experiences)} experiences")

        # 生成行程
        itinerary = "## Personalized Travel Itinerary\n\n"

        # 为每天创建行程
        for day in range(1, days + 1):
            itinerary += f"## Day {day}\n"

            # 分配时间段
            time_slots = ["Morning", "Noon", "Afternoon", "Evening"]
            slot_allocations = {slot: [] for slot in time_slots}

            # 合并所有活动
            all_activities = (
                [("attraction", activity) for activity in selected_attractions] +
                [("food", activity) for activity in selected_foods] +
                [("experience", activity) for activity in selected_experiences]
            )
            random.shuffle(all_activities)

            # 均匀分配到时间段
            for i, (category, activity) in enumerate(all_activities):
                slot = time_slots[i % len(time_slots)]
                slot_allocations[slot].append((category, activity))

            # 构建每天的行程
            for slot in time_slots:
                itinerary += f"### {slot}\n"
                if slot_allocations[slot]:
                    for category, activity in slot_allocations[slot]:
                        itinerary += f"- {category}: {activity['name']}, address: {activity['address']}\n"
                else:
                    itinerary += "- [Placeholder]\n"
                itinerary += "\n"

        return itinerary
    except Exception as e:
        logger.error(f"Error generating itinerary: {str(e)}")
        return f"无法生成行程: {str(e)}"

# FastAPI 应用
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Travel Assistant API"}

@app.get("/generate_itinerary")
async def generate_itinerary_endpoint(
    location: str = Query("Kuching", description="Travel destination"),
    days: int = Query(3, description="Number of days", ge=1),
    food_value: int = Query(2, description="Number of food recommendations (max 5)", ge=0, le=5),
    attraction_value: int = Query(2, description="Number of attraction recommendations (max 5)", ge=0, le=5),
    experience_value: int = Query(1, description="Number of experience recommendations (max 5)", ge=0, le=5)
):
    """
    生成个性化行程，使用查询参数
    - food_value: 食物推荐数量 (0-5)
    - attraction_value: 景点推荐数量 (0-5)
    - experience_value: 体验推荐数量 (0-5)
    """
    try:
        # 调用修改后的 generate_itinerary 函数
        itinerary = generate_itinerary(
            location=location,
            days=days,
            food_count=food_value,
            attraction_count=attraction_value,
            experience_count=experience_value
        )
        return {"itinerary": itinerary, "generated_at": datetime.now().isoformat()}
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in generate_itinerary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating itinerary: {str(e)}")

@app.get("/navigate")
async def navigate_endpoint(start: str = Query(..., description="Starting location"), end: str = Query(..., description="Ending location")):
    """
    导航接口：
    - start: 起点名称或坐标
    - end: 终点名称或坐标
    """
    try:
        # 模拟导航数据
        nav_data = f"Navigate from {start} to {end}: Estimated time 30 mins."
        return {"navigation": nav_data}
    except Exception as e:
        logger.error(f"Error in navigate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in navigation: {str(e)}")

@app.post("/train_bert")
async def train_bert_endpoint(file: UploadFile = File(...)):
    """
    训练 BERT 模型接口（保留文件上传）
    - file: 用户上传的 Excel 偏好文件
    """
    try:
        contents = await file.read()
        # 模拟训练逻辑
        return {"message": "BERT model trained and saved successfully"}
    except Exception as e:
        logger.error(f"Error in train_bert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training BERT: {str(e)}")

@app.post("/train_gpt2")
async def train_gpt2_endpoint():
    """
    训练 GPT-2 模型接口
    """
    try:
        # 模拟训练逻辑
        return {"message": "GPT-2 model trained and saved successfully"}
    except Exception as e:
        logger.error(f"Error in train_gpt2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training GPT-2: {str(e)}")

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)