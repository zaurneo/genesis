import os
import json
import tempfile
import config
from utils.common import save_communication_log


def test_save_communication_log(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        generated_dir = os.path.join(tmpdir, "Generated_Files")
        os.makedirs(generated_dir, exist_ok=True)
        monkeypatch.setattr(config, "GENERATED_FILES_DIR", generated_dir, raising=False)
        monkeypatch.chdir(tmpdir)
        messages = [
            {"speaker": "A", "content": "hello", "timestamp": "0"},
            {"speaker": "B", "content": "hi", "timestamp": "1"},
        ]
        result = save_communication_log(messages)
        assert result.get("success")
        path = os.path.join(generated_dir, "conversation_log.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            lines = [json.loads(line) for line in f]
        assert lines == messages
