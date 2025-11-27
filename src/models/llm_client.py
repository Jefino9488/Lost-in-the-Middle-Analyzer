import os
import json
import time
import requests
import joblib
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from openai import OpenAI
import httpx
from dotenv import load_dotenv

from src.utils import timed_call, ensure_trace

# Initialize joblib memory for caching
memory = joblib.Memory(location=".cache", verbose=0)

@memory.cache
def _cached_ask_dummy(question: str, context: str) -> str:
    # Dummy doesn't really need caching but good for consistency
    if "ANSWER-" in context:
        import re
        match = re.search(r"ANSWER-(\d{4})", context)
        if match:
            return f"The answer is ANSWER-{match.group(1)}"
    return "I couldn't find the answer."

@memory.cache
def _cached_ask_ollama(model_name: str, base_url: str, question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"

@memory.cache
def _cached_ask_gemini(model_name: str, api_key: str, question: str, context: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

@memory.cache
def _cached_ask_openrouter(model_name: str, api_key: str, question: str, context: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


class DummyModel:
    """Fast no-API model that extracts ANSWER-#### tokens from context for dev."""
    provider_key = "dummy-local"
    @staticmethod
    def _extract(context_and_question: str) -> str:
        # Use cached function for extraction logic if needed, but here we just call the cached wrapper
        return _cached_ask_dummy(context_and_question, context_and_question)

    def ask(self, question: str, context: str) -> str:
        return _cached_ask_dummy(question, context)

    def ask_with_trace(self, question: str, context: str):
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        # We wrap the cached call in timed_call to still measure "latency" (even if cached)
        # But for Dummy, latency is negligible.
        res, latency_ms = timed_call(self.ask, question, context)
        return ensure_trace(
            text=res,
            prompt=prompt,
            provider=self.provider_key,
            model="dummy-echo",
            latency_ms=latency_ms,
        )

class OllamaModel:
    def __init__(self, model_name: str = "tinyllama:latest"):
        self.model_name = model_name

    def ask(self, question: str, context: str) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        return _cached_ask_ollama(self.model_name, base_url, question, context)


def list_ollama_models():
    """
    Returns a list of available local models from Ollama.
    - If OLLAMA_BASE_URL/OLLAMA_HOST is set, query the HTTP API (preferred in Docker Compose).
    - Otherwise, fall back to the local `ollama list` CLI.
    """
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
    if base_url:
        try:
            url = base_url.rstrip("/") + "/api/tags"
            resp = httpx.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json() or {}
            models = data.get("models") or data.get("items") or []
            names = []
            for m in models:
                name = m.get("name") or m.get("model") or ""
                if name:
                    names.append(name)
            return names or ["No models found via Ollama HTTP API."]
        except Exception as e:
            return [f"Ollama HTTP error: {e}"]

    # CLI fallback
    import subprocess
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout
        lines = output.strip().split('\n')[1:]

        if not lines:
            return ["No models found via Ollama CLI."]
        model_names = [line.split()[0] for line in lines]

        return model_names

    except FileNotFoundError:
        return ["'ollama' command not found. Is Ollama installed and in your system's PATH?"]
    except subprocess.CalledProcessError as e:
        return [f"Ollama CLI error: {e.stderr.strip()}"]
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]

class GeminiAPIModel:
    provider_key = "gemini"
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name or "gemini-2.0-flash"
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")

    def ask(self, question: str, context: str) -> str:
        return _cached_ask_gemini(self.model_name, self.api_key, question, context)

    def ask_with_trace(self, question: str, context: str):
        prompt_text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with only the exact code if present."
        # Wrap cached call
        res, latency_ms = timed_call(self.ask, question, context)
        return ensure_trace(
            text=res,
            prompt=prompt_text,
            provider=self.provider_key,
            model=self.model_name,
            latency_ms=latency_ms,
        )

class OpenRouterModule:
    provider_key = "openrouter"
    def __init__(self, model_name: str = "google/gemini-pro"):
        self.model_name = model_name
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    def ask(self, question: str, context: str) -> str:
        return _cached_ask_openrouter(self.model_name, self._api_key, question, context)

    def ask_with_trace(self, question: str, context: str):
        # Wrap cached call
        res, latency_ms = timed_call(self.ask, question, context)
        return ensure_trace(
            text=res,
            prompt=f"Context: {context}\n\nQuestion: {question}",
            provider=self.provider_key,
            model=self.model_name,
            latency_ms=latency_ms,
        )
def list_openrouter_models(free_only: bool = True) -> list[str]:
    """
    Returns a list of available models from OpenRouter.
    """
    import httpx
    url = "https://openrouter.ai/api/v1/models"

    try:
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()
            models = response.json().get('data', [])

            model_names = []
            for m in models:
                model_id = m.get('id', '')
                if not model_id:
                    continue

                if free_only:
                    pricing = m.get('pricing', {})
                    input_cost = pricing.get('input')
                    output_cost = pricing.get('output')

                    if (model_id.endswith(':free') or
                            (input_cost in ('0', 0, '0.0') and output_cost in ('0', 0, '0.0'))):
                        model_names.append(model_id)
                else:
                    model_names.append(model_id)

            if not model_names:
                return [("No free models found." if free_only
                         else "No models found or API structure changed.")]

            return model_names

    except httpx.HTTPStatusError as e:
        return [f"[OpenRouter API HTTP error: {e.response.status_code} - {e.response.text}]"]
    except Exception as e:
        return [f"[OpenRouter API error: {e}]"]


def get_model(provider: str, model_name: str = None):
    provider = (provider or "dummy-local").lower()
    if provider == "gemini-api":
        from dotenv import load_dotenv
        load_dotenv()
        return GeminiAPIModel(model_name or "gemini-2.0-flash")
    if provider == "ollama":
        return OllamaModel(model_name or "qwen2:0.5b")
    if provider == "openrouter":
        from dotenv import load_dotenv
        load_dotenv()
        return OpenRouterModule(model_name or "google/gemini-pro")
    return DummyModel()
