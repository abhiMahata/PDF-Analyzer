import os
import requests
from dotenv import load_dotenv




load_dotenv(override=True)
API_KEY = os.getenv("GROQ_API_KEY")




headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
data = response.json()
if "data" in data:
    for model in data["data"]:
        print(model["id"])
else:
    print(data)
