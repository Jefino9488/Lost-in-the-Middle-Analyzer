import google.generativeai as genai
import os
import httpx
from dotenv import load_dotenv


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


def list_ollama_models():
    """
    Returns a list of available local models from Ollama by calling the CLI.
    """
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
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the GeminiAPIModel with a specific model name.
        """
        import os
        from google import genai
        self.model_name = model_name or "gemini-2.0-flash"
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self._client = genai.Client(api_key=api_key)
        except Exception as e:
            raise RuntimeError("google-genai package not installed or GOOGLE_API_KEY missing.") from e

    def ask(self, question: str, context: str) -> str:
        """
        Sends a request to the Gemini model and returns the response.
        """
        from google.genai import types
        try:
            prompt_content = [
                types.Content(
                    role='user',
                    parts=[
                        types.Part.from_text(text=f"Context:\n{context}\n\nQuestion: {question}"),
                        types.Part.from_text(text="Answer with only the exact code if present.")
                    ]
                )
            ]

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip() or ""
        except Exception as e:
            return f"[Gemini API error: {e}]"



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
    """
    A client for the OpenRouter chat completions API using httpx.
    """

    def __init__(self, model_name: str = "google/gemini-pro"):
        # Load environment variables once during initialization
        load_dotenv()

        self.model_name = model_name
        self._api_key = os.getenv("OPENROUTER_API_KEY")

        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. "
                             "Please check your .env file or environment variables.")

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "Lost-in-the-Middle Analyzer",
        }

        self._client = httpx.Client(base_url="https://openrouter.ai/api/v1")

    def ask(self, question: str, context: str) -> str:
        """
        Sends a question and context to the specified model on OpenRouter and returns the response.
        """
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user",
                 "content": f"Context: {context}\n\nQuestion: {question}\nAnswer with only the exact code if present."}
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }

        try:
            response = self._client.post("/chat/completions", headers=self._headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            return f"[OpenRouter API HTTP error: {e.response.status_code} - {e.response.text}]"
        except Exception as e:
            return f"[OpenRouter API error: {e}]"
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
        return OpenRouterModule(model_name)
    return DummyModel()
