"""Async ChatLLM wrapper supporting OpenAI-compatible APIs (GPT-4o-mini, local Qwen, etc.)."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp


class ChatLLM:
    """Async-batched LLM client for OpenAI-compatible endpoints.

    base_url: e.g. "https://runapi.co/v1" — /chat/completions is appended automatically
              OR the full URL ending in /chat/completions
    """

    def __init__(
        self,
        base_url: str,
        key: str,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        # Normalise base_url: strip trailing slash, append /chat/completions if absent
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            base_url = base_url + "/chat/completions"
        self.base_url = base_url
        self.key = key[0] if isinstance(key, tuple) else key
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _async_chat(self, session: aiohttp.ClientSession, req: Dict) -> str:
        prompt = self.prompt.format(**req)
        images: Optional[List[str]] = req.get("images")
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "model": self.model,
            "stream": False,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        for attempt in range(3):
            response_json = None
            try:
                async with session.post(
                    url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    response_json = await response.json()
                    return response_json["choices"][0]["message"]["content"]
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(30)
                else:
                    raise Exception(response_json or str(exc))

    async def _async_chat_batch(self, reqs: List[Dict]) -> List[str]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._async_chat(session, req) for req in reqs]
            return await asyncio.gather(*tasks)

    def chat_batch(self, requests: List[Dict]) -> List[str]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_chat_batch(requests))
        finally:
            loop.close()

    async def _async_chat_single(self, system: str, user: str) -> str:
        """Single synchronous-style call with explicit system/user messages."""
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "model": self.model,
            "stream": False,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                response_json = None
                try:
                    async with session.post(
                        url=self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.key}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps(payload),
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as response:
                        response_json = await response.json()
                        return response_json["choices"][0]["message"]["content"]
                except Exception as exc:
                    if attempt < 2:
                        await asyncio.sleep(30)
                    else:
                        raise Exception(response_json or str(exc))

    def chat_single(self, system: str, user: str) -> str:
        """Make one LLM call with explicit system + user messages. Returns response string."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_chat_single(system, user))
        finally:
            loop.close()
