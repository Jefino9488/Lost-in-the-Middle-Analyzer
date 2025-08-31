import os


class DummyModel:
    """Fast no-API model that extracts ANSWER-#### tokens from context for dev."""
    @staticmethod
    def ask(question: str, context: str) -> str:
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


class GeminiAPIModel:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name or "gemini-1.5-pro"
        try:
            import google.generativeai as genai
            self._client = genai
            self._client.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        except Exception as e:
            raise RuntimeError("google-generativeai package not installed or GOOGLE_API_KEY missing.") from e

    def ask(self, question: str, context: str) -> str:
        try:
            model = self._client.GenerativeModel(self.model_name)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with only the exact code if present."

            response = model.generate_content(
                prompt,
                generation_config=self._client.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip() or ""
        except Exception as e:
            return f"[Gemini API error: {e}]"


def get_model(provider: str, model_name: str = None):
    provider = (provider or "dummy-local").lower()
    if provider == "gemini-api":
        return GeminiAPIModel(model_name or "gemini-1.5-pro")
    if provider == "ollama":
        return OllamaModel(model_name or "qwen2:0.5b")
    return DummyModel()
