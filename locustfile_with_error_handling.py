import logging
import time
from typing import Any, Dict
import httpx
from locust import HttpUser, task, between

from config import OLLAMA_KEY


logger = logging.getLogger(__name__)


def _query_with_retry(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous, low-latency 503-only retry.
    Max attempts: 2
    Backoff: 100 ms
    """

    timeout = httpx.Timeout(30.0, connect=5.0)

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        try:
            response = client.post(
                "/query",
                json=payload,
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
            )

            response.raise_for_status()
            return response.json()
        
        except httpx.ReadTimeout as exc:
            logger.error(f"Request timed out: {exc}")
            raise

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in [503, 504]:
                logger.warning(f"Received {exc.response.status_code} from /query, retrying once")
                time.sleep(0.1)
                
                retry_response = client.post(
                    "/query",
                    json=payload,
                    headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
                )

                if retry_response.status_code == 503:
                    return {
                        "error": "The chatbot service is temporarily unavailable (503). Please try again shortly."
                    }

                retry_response.raise_for_status()
                return retry_response.json()

            else:
                raise


class ChatUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:5000"

    @task
    def send_message(self):
        payload = {
            "query": "hello",
            "history": [],
            "unrelated_streak": 0,
        }

        _ = _query_with_retry(self.host, payload)

# locust -f locustfile_with_error_handling.py --host=http://localhost:5000
