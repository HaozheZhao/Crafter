"""OpenRouter VLM judge call (OpenAI-compatible).

The API key is read from the environment and is never stored in the repo:

    export OPENROUTER_API_KEY="sk-or-..."

The default judge model is ``google/gemini-3.5-flash``. It is a reasoning model
that can spend much of its token budget on hidden reasoning, so ``max_tokens``
is kept generous (override with ``CB_MAX_TOKENS``) or the content can come back
empty.
"""
from __future__ import annotations
import json, os, time
import requests

BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("CB_API_KEY")
    if not key:
        raise RuntimeError(
            "Set the OPENROUTER_API_KEY environment variable to your OpenRouter key.")
    return key


try:
    import json_repair
    def _loads(s): return json_repair.loads(s)
except ImportError:
    def _loads(s): return json.loads(s)


def call_judge(system: str, user_content, model: str = "google/gemini-3.5-flash",
               max_tokens: int = 16000, temperature: float = 0.0,
               want_json: bool = True, max_retries: int = 8,
               timeout: int = 240, seed: int | None = None) -> dict:
    """Run one judge call. ``user_content`` is a list of OpenAI content parts.

    Returns the parsed JSON dict on success, else ``{"_error": msg}``.
    """
    max_tokens = int(os.environ.get("CB_MAX_TOKENS", max_tokens))
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if want_json:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    last = ""
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{BASE}/chat/completions", headers=headers,
                              json=body, timeout=timeout)
            if r.status_code == 200:
                txt = r.json()["choices"][0]["message"]["content"] or ""
                clean = txt.replace("```json", "").replace("```", "").strip()
                if not clean:
                    last = "empty content (raise CB_MAX_TOKENS?)"
                else:
                    try:
                        d = _loads(clean)
                        if isinstance(d, dict):
                            return d
                        last = f"non-dict json: {type(d).__name__}"
                    except Exception as e:
                        last = f"json parse: {e}"
                        if attempt == max_retries - 1:
                            return {"_error": last}
            else:
                last = f"status={r.status_code} body={r.text[:160]}"
                if r.status_code in (429, 500, 502, 503, 529):
                    time.sleep(5 + attempt * 8)
                    continue
        except Exception as e:
            last = f"exc {type(e).__name__}: {e}"
        if attempt < max_retries - 1:
            time.sleep(4 + attempt * 5)
    return {"_error": last[:300]}
