require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
const PORT = 4000;

// 允许跨域
app.use(cors());
app.use(express.json());

// ✅ 配置 Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-pro" });

// 📍 1. 解析用户输入，获取起点 & 终点 GPS 坐标
async function getCoordinates(location) {
  const prompt = `请给我 "${location}" 的经纬度（格式：纬度, 经度）。`;

  try {
    const response = await model.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] });
    const textResponse = response.response.text(); // ✅ 修正获取文本的方法
    const coordinates = textResponse.match(/([-+]?\d*\.\d+),\s*([-+]?\d*\.\d+)/);

    if (coordinates) {
      return { lat: parseFloat(coordinates[1]), lng: parseFloat(coordinates[2]) };
    } else {
      throw new Error("无法解析地点坐标");
    }
  } catch (error) {
    console.error("❌ 获取经纬度失败:", error);
    return null;
  }
}

// 📍 2. 获取 Waze 最佳路线（✅ 需要用 Waze API，不是网页 URL）
async function getWazeRoute(startLat, startLng, endLat, endLng) {
  const WAZE_API_KEY = process.env.WAZE_API_KEY;
  const url = `https://www.waze.com/RoutingManager/routingRequest?from=x:${startLng} y:${startLat}&to=x:${endLng} y:${endLat}&returnJSON=true&key=${WAZE_API_KEY}`;

  try {
    const response = await axios.get(url);
    return response.data;
  } catch (error) {
    console.error("❌ 获取 Waze 路线失败:", error);
    return null;
  }
}

// 🌍 API 路由：获取最佳旅游路线
app.post("/get-route", async (req, res) => {
  const { start, destination } = req.body;

  if (!start || !destination) {
    return res.status(400).json({ error: "请输入起点和终点" });
  }

  try {
    console.log(`📍 正在解析地点: ${start} -> ${destination}`);

    const startCoords = await getCoordinates(start);
    const endCoords = await getCoordinates(destination);

    if (!startCoords || !endCoords) {
      return res.status(500).json({ error: "获取经纬度失败" });
    }

    console.log(`🚀 计算最佳路线: ${startCoords.lat},${startCoords.lng} -> ${endCoords.lat},${endCoords.lng}`);
    
    const route = await getWazeRoute(startCoords.lat, startCoords.lng, endCoords.lat, endCoords.lng);

    if (!route) {
      return res.status(500).json({ error: "无法获取路线数据" });
    }

    res.json({ start: startCoords, destination: endCoords, route });
  } catch (error) {
    console.error("❌ 处理请求失败:", error);
    res.status(500).json({ error: "服务器错误" });
  }
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`🚀 服务器运行在 http://localhost:${PORT}`);
});

app.get("/", (req, res) => {
    res.send("欢迎使用 AI 旅行助手！");
    });
