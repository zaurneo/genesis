from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from config import gpt_api_key, claude_api_key

model_client_gpt4o = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    api_key=gpt_api_key
)

model_client_claude3s = AnthropicChatCompletionClient(
    model="claude-3-7-sonnet-20250219",
    api_key=claude_api_key
)