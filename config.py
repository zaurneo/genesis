import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GENERATED_FILES_DIR = os.path.join(BASE_DIR, "working")
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

WORK_DIR = os.path.join(BASE_DIR, "working")
os.makedirs(WORK_DIR, exist_ok=True)

# Directories for domain specific knowledge. These can be overridden with
# environment variables to point at custom locations. The defaults live inside
# the repository ``knowledge`` folder.
DOMAIN_KNOWLEDGE_DOCS_DIR = os.environ.get(
    "DOMAIN_KNOWLEDGE_DOCS_DIR",
    os.path.join(BASE_DIR, "knowledge", "input"),
)
DOMAIN_KNOWLEDGE_STORAGE_DIR = os.environ.get(
    "DOMAIN_KNOWLEDGE_STORAGE_DIR",
    os.path.join(BASE_DIR, "knowledge", "output"),
)

