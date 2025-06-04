from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
import asyncio
import openai
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file



class OpenAIChatCompletionClientWithRetry(OpenAIChatCompletionClient):
    """OpenAI client with basic exponential backoff for rate limit errors."""

    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

    async def create(self, *args, **kwargs):  # type: ignore[override]
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                return await super().create(*args, **kwargs)
            except openai.RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

model_client_gpt4o = OpenAIChatCompletionClientWithRetry(
    model="gpt-4o-2024-08-06",
    api_key=os.environ["gpt_api_key"]
)

model_client_claude3s = AnthropicChatCompletionClient(
    model="claude-3-7-sonnet-20250219",
    api_key=os.environ["claude_api_key"] 
)

