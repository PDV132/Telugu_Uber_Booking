from langchain.schema import LLMResult
from typing import Any, List

class MockLLM:
    """A safe mock LLM for simulation without calling OpenAI."""
    
    # Minimal async predict interface
    async def apredict(self, prompt: str, **kwargs) -> str:
        print(f"[MOCK LLM] Prompt received: {prompt}")
        return "Simulated response for testing"

    async def agenerate(self, prompts: List[Any], **kwargs) -> LLMResult:
        print(f"[MOCK LLM] Generating for prompts: {prompts}")
        return LLMResult(generations=[{"text": "Simulated response for testing"}])
