import asyncio
import logging
from typing import Any, Dict

import httpx
from locust import HttpUser, task, between

from config import OLLAMA_KEY


logger = logging.getLogger(__name__)

# Small, configurable concurrency limit for outbound calls
_MAX_CONCURRENT_REQUESTS = 5
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


async def _query_with_retry(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the chatbot `/query` endpoint with minimal 503-only retries.

    - Max attempts: 2 (1 initial + 1 retry)
    - Backoff: fixed 100 ms, no jitter, no exponential backoff.
    - Timeouts: connect <= 1s, read <= 5s.
    - Retry condition: only on httpx.HTTPStatusError with status code == 503.
    - On second 503: fail fast and return a clear, user-safe error payload.
    """

    timeout = httpx.Timeout(5.0, connect=1.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        attempt = 0
        while True:
            attempt += 1
            try:
                async with _semaphore:
                    response = await client.post(
                        "/query",
                        json=payload,
                        headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
                    )

                # Only raise_for_status so we can handle HTTPStatusError
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as exc:
                # Only retry on 503; everything else bubbles up immediately
                if exc.response is not None and exc.response.status_code == 503:
                    if attempt == 1:
                        # Single, clear warning for observability
                        logger.warning("Received 503 from /query, retrying once")
                        await asyncio.sleep(0.1)
                        continue

                    # Second 503: fail fast with a user-safe error payload
                    return {
                        "error": "The chatbot service is temporarily unavailable (503). Please try again shortly."
                    }

                # Non-503 errors are not retried or wrapped
                raise


class ChatUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:5000"  # adjust if different

    @task
    def send_message(self):
        payload = {
            "query": "Hello from Locust",
            "history": [],
            "unrelated_streak": 0,
        }

        # Run the async HTTPX client with bounded concurrency and 503-only retry
        try:
            _ = asyncio.run(_query_with_retry(self.host, payload))
        except httpx.HTTPError as exc:
            # For load testing we just record that an error happened; we do not retry
            # on anything other than the explicit 503 path handled above.
            logger.error("HTTP error calling /query: %s", exc)

# locust -f locustfile.py --host=http://localhost:5000
