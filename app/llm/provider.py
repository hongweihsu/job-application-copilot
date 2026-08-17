from time import perf_counter
from typing import Protocol

from openai import OpenAI

from app.llm.config import LLMConfig
from app.llm.models import LLMRequirementDecision
from app.llm.prompt import SYSTEM_PROMPT, build_user_prompt


class RequirementDecisionProvider(Protocol):
    model_name: str

    def decide(
        self, requirement: str, evidence_units: list[tuple[str, str]]
    ) -> LLMRequirementDecision: ...


class OpenAIRequirementDecisionProvider:
    def __init__(self, config: LLMConfig):
        self.model_name = config.model
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.elapsed_seconds = 0.0
        self._client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def decide(
        self, requirement: str, evidence_units: list[tuple[str, str]]
    ) -> LLMRequirementDecision:
        started = perf_counter()
        try:
            response = self._client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(requirement, evidence_units)},
                ],
                text_format=LLMRequirementDecision,
            )
        finally:
            self.elapsed_seconds += perf_counter() - started
        self.request_count += 1
        if response.usage is not None:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("The model did not return a parsed requirement decision")
        return parsed
