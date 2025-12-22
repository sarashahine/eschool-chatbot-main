from locust import HttpUser, task, between

from config import OLLAMA_KEY

class ChatUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:5000"  # adjust if different

    @task
    def send_message(self):
        self.client.post(
            "/query",
            json={
                "query": "Hello from Locust",
                "history": [],
                "unrelated_streak": 0
            },
            headers={
                "Authorization": f"Bearer {OLLAMA_KEY}"
            }
        )

# locust -f locustfile.py --host=http://localhost:5000
