import json
import os
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
        if path:
            url = self.config.api_url.rstrip("/") + "/" + path.lstrip("/")
        else:
            url = self.config.api_url
        
        headers = {
            "Content-Type": "application/json",
        }
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["api-key"] = api_key
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
                if isinstance(error_json, dict) and "error" in error_json:
                    err = error_json["error"]
                    if isinstance(err, dict) and "message" in err:
                        detail = err["message"]
                    else:
                        detail = str(err)
                else:
                    detail = error_json.get("detail", error_body)
            except Exception:
                detail = error_body
            raise RuntimeError(f"HTTP Error {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach API server at {url}: {e.reason}") from e

    def _get(self, path: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Send a HTTP GET request to the API."""
        if path:
            url = self.config.api_url.rstrip("/") + "/" + path.lstrip("/")
        else:
            url = self.config.api_url
        
        headers = {}
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["api-key"] = api_key
        if self.config.headers:
            headers.update(self.config.headers)

        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_data = response.read().decode("utf-8")
                return json.loads(resp_data)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_json = json.loads(error_body)
                if isinstance(error_json, dict) and "error" in error_json:
                    err = error_json["error"]
                    if isinstance(err, dict) and "message" in err:
                        detail = err["message"]
                    else:
                        detail = str(err)
                else:
                    detail = error_json.get("detail", error_body)
            except Exception:
                detail = error_body
            raise RuntimeError(f"HTTP Error {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach API server at {url}: {e.reason}") from e

    def _extract_text_from_output(self, output_list: List[Dict[str, Any]]) -> str:
        """Extract textual content from Responses API output blocks."""
        parts = []
        for item in output_list:
            if not isinstance(item, dict):
                continue
            
            item_type = item.get("type")
            
            # Case 1: type is text
            if item_type == "text":
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                    
            # Case 2: type is output_text
            elif item_type == "output_text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    
            # Case 3: type is message
            elif item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for sub_item in content:
                        if not isinstance(sub_item, dict):
                            continue
                        sub_type = sub_item.get("type")
                        if sub_type in ("text", "output_text"):
                            sub_content = sub_item.get("content") or sub_item.get("text")
                            if isinstance(sub_content, str):
                                parts.append(sub_content)
                        elif "text" in sub_item and isinstance(sub_item["text"], str):
                            parts.append(sub_item["text"])
                            
            # General fallback if there is a 'content' or 'text' field directly
            elif "content" in item and isinstance(item["content"], str):
                parts.append(item["content"])
            elif "text" in item and isinstance(item["text"], str):
                parts.append(item["text"])
                
        return "".join(parts)

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        steering_override: Optional[Dict[str, float]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None
    ) -> str:
        # Determine active steering vector (overrides take precedence)
        steering = steering_override if steering_override is not None else self.config.steering
        
        is_responses_api = "responses" in self.config.api_url.lower()

        payload = {
            "stream": False,
        }

        temp = temperature_override if temperature_override is not None else self.config.temperature
        if temp is not None:
            payload["temperature"] = temp

        if is_responses_api:
            payload["input"] = messages
            payload["max_output_tokens"] = max_tokens_override if max_tokens_override is not None else self.config.max_tokens
        else:
            payload["messages"] = messages
            payload["max_tokens"] = max_tokens_override if max_tokens_override is not None else self.config.max_tokens

        if self.config.model_name:
            payload["model"] = self.config.model_name

        if steering is not None:
            payload["steering"] = steering

        if is_responses_api:
            response = self._post("", payload)
        else:
            response = self._post("chat/completions", payload)

        if isinstance(response, dict) and response.get("error") is not None:
            error_obj = response["error"]
            if isinstance(error_obj, dict) and "message" in error_obj:
                error_msg = error_obj["message"]
            else:
                error_msg = str(error_obj)
            raise RuntimeError(f"API Error: {error_msg}")

        if is_responses_api:
            
            if "output" in response:
                text = self._extract_text_from_output(response["output"])
                if text:
                    return text
            
            # Fallback to general parsing if output is missing or empty
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
            raise RuntimeError(f"Malformed Responses API response format: {response}")
        else:
            # Note: _post was already called above
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                # Try fallback parsing if they used chat/completions but got responses-style format
                if "output" in response:
                    text = self._extract_text_from_output(response["output"])
                    if text:
                        return text
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
        if temperature is None:
            temperature = 0.0

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
