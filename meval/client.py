import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from meval.schemas import ModelConfig

class LLMBackend(ABC):
    """Abstract interface for large language model backends."""
    
    @abstractmethod
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        steering_override: Optional[Dict[str, float]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None
    ) -> str:
        """Generate a response text given conversation messages."""
        pass


class APIBackend(LLMBackend):
    """API-based backend that queries an OpenAI-compatible endpoint."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def _post(self, path: str, payload: Dict[str, Any], timeout: float = 180.0) -> Dict[str, Any]:
        """Send a HTTP POST request to the API."""
        url = self.config.api_url.rstrip("/") + "/" + path.lstrip("/")
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.headers:
            headers.update(self.config.headers)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_data = response.read().decode("utf-8")
                return json.loads(resp_data)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_json = json.loads(error_body)
                detail = error_json.get("detail", error_body)
            except Exception:
                detail = error_body
            raise RuntimeError(f"HTTP Error {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach API server at {url}: {e.reason}") from e

    def _get(self, path: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Send a HTTP GET request to the API."""
        url = self.config.api_url.rstrip("/") + "/" + path.lstrip("/")
        
        headers = {}
        if self.config.headers:
            headers.update(self.config.headers)

        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_data = response.read().decode("utf-8")
                return json.loads(resp_data)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP Error {e.code} from {url}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach API server at {url}: {e.reason}") from e

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        steering_override: Optional[Dict[str, float]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None
    ) -> str:
        # Determine active steering vector (overrides take precedence)
        steering = steering_override if steering_override is not None else self.config.steering
        
        payload = {
            "messages": messages,
            "temperature": temperature_override if temperature_override is not None else self.config.temperature,
            "max_tokens": max_tokens_override if max_tokens_override is not None else self.config.max_tokens,
            "stream": False,
        }

        if self.config.model_name:
            payload["model"] = self.config.model_name

        if steering is not None:
            payload["steering"] = steering

        response = self._post("chat/completions", payload)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Malformed response format from API: {response}") from e

    def list_models(self) -> List[Dict[str, Any]]:
        """List loaded models from /v1/models endpoint."""
        try:
            res = self._get("models")
            return res.get("data", [])
        except Exception:
            res = self._get("v1/models")
            return res.get("data", [])

    def get_models_info(self) -> Dict[str, Any]:
        """Fetch model info from /models/info (specific to steerable-model-runner)."""
        return self._get("models/info")


class HFBackend(LLMBackend):
    """Local backend that loads and runs a model directly via Hugging Face transformers."""

    def __init__(self, config: ModelConfig):
        self.config = config

        # Lazy load to avoid importing heavy frameworks for pure API backend runs
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Direct Hugging Face backend requires 'torch' and 'transformers' libraries. "
                "Please run: pip install torch transformers accelerate"
            ) from e

        print(f"Loading local Hugging Face model: {config.model_name}...")
        
        dtype_str = config.torch_dtype.lower().strip()
        if dtype_str in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif dtype_str in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif dtype_str == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32
            print(f"Warning: Unknown torch_dtype '{dtype_str}'. Defaulting to torch.float32.")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        steering_override: Optional[Dict[str, float]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None
    ) -> str:
        import torch

        if steering_override is not None:
            print("Warning: Direct HFBackend does not support steerable LoRA hooks natively. Steering ignored.")

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        max_tokens = max_tokens_override if max_tokens_override is not None else self.config.max_tokens
        temperature = temperature_override if temperature_override is not None else self.config.temperature

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        input_len = input_ids.shape[1]
        response_ids = outputs[0][input_len:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)


# Keep ModelRunnerClient as alias for backward compatibility / API operations
ModelRunnerClient = APIBackend

def get_backend(config: ModelConfig) -> LLMBackend:
    """Helper factory to instantiate the appropriate backend based on config."""
    if config.backend.lower().strip() == "hf":
        return HFBackend(config)
    else:
        return APIBackend(config)
