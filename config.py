import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR = os.path.join(BASE_DIR, "Generated_Files")
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

gpt_api_key = ""

claude_api_key = ""