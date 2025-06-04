import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GENERATED_FILES_DIR = os.path.join(BASE_DIR, "generated_files")
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

WORK_DIR = os.path.join(BASE_DIR, "working_folder")
os.makedirs(WORK_DIR, exist_ok=True)

DOMAIN_KNOWLEDGE_DOCS_DIR = os.path.join(BASE_DIR, "domain_knowledge_input")
os.makedirs(DOMAIN_KNOWLEDGE_DOCS_DIR, exist_ok=True)

DOMAIN_KNOWLEDGE_STORAGE_DIR = os.path.join(BASE_DIR, "domain_knowledge_storage")
os.makedirs(DOMAIN_KNOWLEDGE_STORAGE_DIR, exist_ok=True)

COMM_DIR = os.path.join(BASE_DIR, "common_directory")
os.makedirs(COMM_DIR, exist_ok=True)


