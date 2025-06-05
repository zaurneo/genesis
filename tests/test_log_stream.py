import os
import json
import tempfile
import asyncio
import config
from utils.common import log_stream

class DummyStream:
    def __init__(self, messages):
        self.messages = messages
    def __aiter__(self):
        self._iter = iter(self.messages)
        return self
    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

class DummyTeam:
    def __init__(self, messages):
        self._messages = messages
    def run_stream(self, task=None):
        return DummyStream(self._messages)

def test_log_stream(monkeypatch):
    messages = [
        {"speaker": "A", "content": "hi"},
        {"speaker": "B", "content": "bye"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_dir = os.path.join(tmpdir, "Generated_Files")
        os.makedirs(gen_dir, exist_ok=True)
        monkeypatch.setattr(config, "GENERATED_FILES_DIR", gen_dir, raising=False)
        monkeypatch.chdir(tmpdir)
        team = DummyTeam(messages)
        async def run():
            stream = log_stream(team.run_stream("task"))
            async for _ in stream:
                pass
        asyncio.run(run())
        path = os.path.join(gen_dir, "conversation_log.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            lines = [json.loads(line) for line in f]
        assert lines == messages
