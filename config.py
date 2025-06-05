import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GENERATED_FILES_DIR = os.path.join(BASE_DIR, "working")
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

WORK_DIR = os.path.join(BASE_DIR, "working")
os.makedirs(WORK_DIR, exist_ok=True)

