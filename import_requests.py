import requests

url = "http://127.0.0.1:8000/train_bert"
files = {"file": open(r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\user_preferences.xlsx", "rb")}
params = {
    "location": "Kuching",
    "days": 3,
    "food_value": 2,
    "attraction_value": 2,
    "experience_value": 1
}
response = requests.post(url, files=files, params=params)
print(response.text)
