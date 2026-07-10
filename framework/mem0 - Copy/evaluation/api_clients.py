import os
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI


def get_env_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    value = value.strip() if value else ""
    return value or None


def get_chat_model(default: str = "gpt-4o-mini") -> str:
    return get_env_value("MODEL") or default


def get_judge_model(default: str = "gpt-4o-mini") -> str:
    return get_env_value("JUDGE_MODEL") or get_env_value("MODEL") or default


def get_embedding_model(default: str = "text-embedding-3-small") -> str:
    return get_env_value("EMBEDDING_MODEL") or default


def get_chat_api_values() -> Tuple[str, Optional[str], Dict[str, Any]]:
    beeknoee_api_key = get_env_value("BEEKNOEE_API_KEY")
    beeknoee_base_url = get_env_value("BEEKNOEE_BASE_URL")
    if not beeknoee_api_key:
        raise RuntimeError("BEEKNOEE_API_KEY is required for chat completions; OPENAI_API_KEY is used only for embeddings.")
    if not beeknoee_base_url:
        raise RuntimeError("BEEKNOEE_BASE_URL is required for chat completions.")
    return beeknoee_api_key, beeknoee_base_url, {
        "provider": "beeknoee",
        "uses_beeknoee_api_key": True,
        "uses_openai_api_key_for_chat": False,
        "base_url_configured": True,
    }


def create_chat_client() -> Tuple[OpenAI, Dict[str, Any]]:
    api_key, base_url, info = get_chat_api_values()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), info


def create_embedding_client() -> Tuple[OpenAI, Dict[str, Any]]:
    api_key = get_env_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
    return OpenAI(api_key=api_key), {
        "provider": "openai",
        "uses_openai_api_key": True,
        "base_url_configured": False,
    }


def build_mem0_llm_config(model: str, *, temperature: float = 0, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    api_key, base_url, _ = get_chat_api_values()
    config = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
    }
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    if base_url:
        config["openai_base_url"] = base_url
    return config


def build_mem0_embedding_config(model: str) -> Dict[str, Any]:
    api_key = get_env_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
    return {
        "model": model,
        "api_key": api_key,
        "openai_base_url": "https://api.openai.com/v1",
    }
