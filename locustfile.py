from locust import HttpUser, task, between
import json

from config import OLLAMA_KEY

class MyUser(HttpUser):
    wait_time = between(5, 10)

    @task
    def post_query(self):
        payload = {
            "query": "what is eschool?",
            "history": [],
            "unrelated_streak": 0
        }

        response = self.client.post(
            "/query",
            json=payload,
            headers={"Authorization": f"Bearer {OLLAMA_KEY}"})
        
        print(f"Status code: {response.status_code}")
        print(f"Response JSON: {response.json()}")

# locust -f locustfile.py --host=http://localhost:5000