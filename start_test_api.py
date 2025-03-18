import requests

url = "http://localhost:8800/generate_itinerary?location=Kuching&days=3&food_value=2&attraction_value=2&experience_value=1&use_gpt2=true"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
print(response.text)