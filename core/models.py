import os

import google.generativeai as genai
import httpx
from dotenv import load_dotenv

from core.utils import timed_call, ensure_trace


class DummyModel:
    """Fast no-API model that extracts ANSWER-#### tokens from context for dev."""
    provider_key = "dummy-local"
    @staticmethod
    def _extract(context_and_question: str) -> str:
        import re
        m = re.search(r"ANSWER-\d{4}", context_and_question)
        if m:
            return m.group(0)
        m2 = re.search(r"ANSWER-\d{4}", context_and_question)
        return m2.group(0) if m2 else "I couldn't find the code."

    def ask(self, question: str, context: str) -> str:
        return self._extract(context + "\n" + question)

    def ask_with_trace(self, question: str, context: str):
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        res, latency_ms = timed_call(self._extract, prompt)
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
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
        if base_url:
            try:
                url = base_url.rstrip("/") + "/api/generate"
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                }
                resp = httpx.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                text = data.get("response") or data.get("text") or ""
                return str(text).strip()
            except Exception as e:
                return f"[Ollama HTTP error: {e}]"
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            return result.stdout.decode("utf-8").strip()
        except Exception as e:
            return f"[Ollama error: {e}]"


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
        import os
        from google import genai
        self.model_name = model_name or "gemini-2.0-flash"
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        self._client = genai.Client(api_key=api_key)

    def ask(self, question: str, context: str) -> str:
        return self.ask_with_trace(question, context)["text"]

    def ask_with_trace(self, question: str, context: str):
        from google.genai import types
        prompt_text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with only the exact code if present."
        def _call():
            prompt_content = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)],
                )
            ]
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
            )
            return response.text if hasattr(response, "text") else str(response)

        res, latency_ms = timed_call(_call)
        return ensure_trace(
            text=res,
            prompt=prompt_text,
            provider=self.provider_key,
            model=self.model_name,
            latency_ms=latency_ms,
        )



def list_gemini_models():
    """Return a list of available Gemini/GenAI models that support generateContent.
    """
    try:
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return ["API key not set (set GOOGLE_API_KEY in your .env file)"]

        genai.configure(api_key=api_key)

        models = []
        print("Fetching model list. This may take a moment...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)

        if not models:
            return ["No models found that support generateContent"]

        return models

    except ImportError:
        return [
            "google-generativeai or python-dotenv not installed. Run: pip install -q -U google-generativeai python-dotenv"]
    except Exception as e:
        return [f"Gemini API error: {e}"]


class OpenRouterModule:
    provider_key = "openrouter"
    def __init__(self, model_name: str = "google/gemini-pro"):
        load_dotenv()
        self.model_name = model_name
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "Lost-in-the-Middle Analyzer",
        }
        self._client = httpx.Client(base_url="https://openrouter.ai/api/v1")

    def ask(self, question: str, context: str) -> str:
        return self.ask_with_trace(question, context)["text"]

    def ask_with_trace(self, question: str, context: str):
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user",
                 "content": f"Context: {context}\n\nQuestion: {question}\nAnswer with only the exact code if present."}
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        def _call():
            response = self._client.post("/chat/completions", headers=self._headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        try:
            res, latency_ms = timed_call(_call)
        except Exception as e:
            res, latency_ms = f"[OpenRouter API error: {e}]", 0

        return ensure_trace(
            text=res,
            prompt=data["messages"][0]["content"],
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
