import json
import os
import tempfile
import config
from tools import validate_json_file


def setup_tmpdir(monkeypatch):
    tmpdir = tempfile.TemporaryDirectory()
    generated_dir = os.path.join(tmpdir.name, "Generated_Files")
    os.makedirs(generated_dir, exist_ok=True)
    monkeypatch.setattr(config, "GENERATED_FILES_DIR", generated_dir, raising=False)
    monkeypatch.chdir(tmpdir.name)
    return tmpdir, generated_dir


def test_validate_json_file_success(monkeypatch):
    tmpdir, generated_dir = setup_tmpdir(monkeypatch)
    try:
        path = os.path.join(generated_dir, "sample.json")
        with open(path, "w") as f:
            json.dump({"a": 1, "b": 2}, f)
        result = validate_json_file("sample.json")
        assert result == {"success": True, "file": path}
    finally:
        tmpdir.cleanup()


def test_validate_json_file_invalid(monkeypatch):
    tmpdir, generated_dir = setup_tmpdir(monkeypatch)
    try:
        path = os.path.join(generated_dir, "sample.json")
        with open(path, "w") as f:
            f.write('{"a": 1, "b": 2]')  # malformed JSON
        result = validate_json_file("sample.json")
        assert result["error"] == "Invalid JSON"
        assert result["line"] >= 1
        assert result["column"] >= 1
    finally:
        tmpdir.cleanup()


def test_validate_json_file_trailing(monkeypatch):
    tmpdir, generated_dir = setup_tmpdir(monkeypatch)
    try:
        path = os.path.join(generated_dir, "sample.json")
        with open(path, "w") as f:
            f.write('{"a": 1}')
            f.write('extra')
        result = validate_json_file("sample.json")
        assert result["error"] == "Invalid JSON"
    finally:
        tmpdir.cleanup()
