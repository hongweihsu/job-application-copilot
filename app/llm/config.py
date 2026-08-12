import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )


def llm_analysis_enabled() -> bool:
    return os.getenv("LLM_ANALYSIS_ENABLED", "false").lower() in {"1", "true", "yes"}
