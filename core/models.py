import os
from typing import Optional

class DummyModel:
    """Fast no-API model that extracts ANSWER-#### tokens from context for dev."""
    def ask(self, question: str, context: str) -> str:
        import re
        m = re.search(r"ANSWER-\d{4}", context)
        if m:
            return m.group(0)
        # sometimes the prompt was the question + chunk; try to find in prompt too
        m2 = re.search(r"ANSWER-\d{4}", question)
        return m2.group(0) if m2 else "I couldn't find the code."

class OllamaModel:
    def __init__(self, model_name: str = "tinyllama:latest"):
        self.model_name = model_name

    def ask(self, question: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."
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

def _init_vertex():
    # lazy import
    try:
        from google.cloud import aiplatform
        return aiplatform
    except Exception as e:
        raise RuntimeError("Vertex SDK not installed or not available.") from e

class VertexGemini:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name or "gemini-1.5-flash"
    def ask(self, question: str, context: str) -> str:
        # Note: this is a simple wrapper. Users must provide proper project & auth.
        # Keep prompt short and deterministic (temperature 0) when using in evaluation.
        try:
            from vertexai.language_models import TextGenerationModel
        except Exception:
            # fallback older import
            from vertexai.generative_models import GenerativeModel as GM
            gm = GM(self.model_name)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with the exact code if present."
            resp = gm.generate([prompt])
            return resp.text or ""
        model = TextGenerationModel.from_pretrained(self.model_name)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with the exact code if present."
        out = model.predict(prompt, max_output_tokens=256, temperature=0)
        return getattr(out, "text", str(out)) or ""

class OpenAIModel:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name or "gpt-4o-mini"
        # require OPENAI_API_KEY in env
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self._client = openai
        except Exception as e:
            raise RuntimeError("openai package not installed or OPENAI_API_KEY missing.") from e

    def ask(self, question: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with only the code if present."
        # Using chat completion style for broad compatibility
        resp = self._client.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()

def get_model(provider: str, model_name: str = None):
    provider = (provider or "dummy-local").lower()
    if provider == "vertex-gemini":
        return VertexGemini(model_name or "gemini-1.5-flash")
    if provider == "openai":
        return OpenAIModel(model_name or "gpt-4o-mini")
    if provider == "ollama":
        return OllamaModel(model_name or "qwen2:0.5b")
    return DummyModel()
