
"""
DESCRIPTION: This file contains miscellaneous functions that are used in multiple scripts.
"""

import os
import json
import logging
from typing import List
import config
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()


WORK_DIR = config.WORK_DIR



logger = logging.getLogger(__name__)




# Metadata describing the signature and purpose of each general purpose tool.
agent_functions = [
    {
        "name": "read_file",
        "description": "Reads a file and returns its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": f"Path to the file relative to {WORK_DIR}.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "read_multiple_files",
        "description": "Reads multiple files and returns their contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of paths relative to {WORK_DIR}.",
                },
            },
            "required": ["file_paths"],
        },
    },
    {
        "name": "read_directory_contents",
        "description": "Return file names contained in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": f"Directory relative to {WORK_DIR}.",
                },
            },
            "required": ["directory_path"],
        },
    },
    {
        "name": "save_file",
        "description": "Save a file to disk without overwriting existing files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": f"Destination path relative to {WORK_DIR}.",
                },
                "file_contents": {
                    "type": "string",
                    "description": "Contents to write to the file.",
                },
            },
            "required": ["file_path", "file_contents"],
        },
    },
    {
        "name": "save_multiple_files",
        "description": "Save multiple files in one call without overwriting.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of paths relative to {WORK_DIR}.",
                },
                "file_contents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file contents matching file_paths.",
                },
            },
            "required": ["file_paths", "file_contents"],
        },
    },
    {
        "name": "execute_code_block",
        "description": "Execute a Python or bash code block and capture output.",
        "parameters": {
            "type": "object",
            "properties": {
                "lang": {"type": "string", "description": "Language of the code"},
                "code_block": {
                    "type": "string",
                    "description": "Code block to execute. If first line is '# filename: <name>' it will be saved.",
                },
            },
            "required": ["lang", "code_block"],
        },
    },

]















# =============================================================================
# JSON
# =============================================================================




def load_json(filename):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as file:
            return json.load(file)
    else:
        return []


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_communication_log(
    messages: List[dict], file_name: str = "conversation_log.jsonl"
) -> dict:
    """Save a sequence of chat messages to ``file_name`` in ``GENERATED_FILES_DIR``.

    The log is stored using newline-delimited JSON so that each message can be
    easily parsed while keeping the file size manageable for long runs.

    Parameters
    ----------
    messages:
        A list of dictionaries representing the conversation. Each dictionary
        should include at least ``speaker`` and ``content`` keys, but arbitrary
        additional metadata is allowed.
    file_name:
        Destination file name relative to :data:`GENERATED_FILES_DIR`.

    Returns
    -------
    dict
        Information about the saved log or an error message.
    """
    path = file_path(file_name)
    try:
        with open(path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return {"success": True, "file": path, "messages_saved": len(messages)}
    except Exception as e:  # pragma: no cover - unexpected file errors
        return {"error": f"Failed to save conversation log: {str(e)}"}


async def log_stream(
    stream: 'AsyncIterable[dict]', file_name: str = "conversation_log.jsonl"
) -> 'AsyncIterator[dict]':
    """Yield items from ``stream`` while logging them to ``file_name``.

    Each message from ``stream`` is written to ``file_name`` as newline-delimited
    JSON. The file lives inside :data:`GENERATED_FILES_DIR`.
    """
    path = file_path(file_name)
    with open(path, "w") as f:
        async for msg in stream:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            yield msg





# =============================================================================
# Path / Directory
# =============================================================================

def file_path(name: str) -> str:
    """Return the absolute path for generated files."""
    return os.path.join(config.GENERATED_FILES_DIR, name)


def extract_base_path(full_path, target_directory):
    """
    Extracts the base path up to and including the target directory from the given full path.

    :param full_path: The complete file path.
    :param target_directory: The target directory to which the path should be truncated.
    :return: The base path up to and including the target directory, or None if the target directory is not in the path.
    """
    path_parts = full_path.split(os.sep)
    if target_directory in path_parts:
        target_index = path_parts.index(target_directory)
        base_path = os.sep.join(path_parts[: target_index + 1])
        return base_path
    else:
        return None

def map_directory_to_json(dir_path):
    def dir_to_dict(path):
        dir_dict = {"name": os.path.basename(path)}
        if os.path.isdir(path):
            dir_dict["type"] = "directory"
            dir_dict["children"] = [
                dir_to_dict(os.path.join(path, x)) for x in os.listdir(path)
            ]
        else:
            dir_dict["type"] = "file"
        return dir_dict

    root_structure = dir_to_dict(dir_path)
    return json.dumps(root_structure, indent=4)


def read_directory_contents(directory_path: str) -> List[str]:
    """List all files in ``directory_path`` relative to ``WORK_DIR``."""
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{directory_path}"))
    return os.listdir(resolved_path)




# =============================================================================
# Files
# =============================================================================

def load_file(filename):
    with open(filename) as f:
        file = f.read()

        if len(file) == 0:
            raise ValueError("filename cannot be empty.")

        return file

def read_file(file_path: str) -> str:
    """Return the contents of ``file_path`` relative to ``WORK_DIR``."""
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    with open(resolved_path, "r") as f:
        return f.read()


def read_multiple_files(file_paths: List[str]) -> List[str]:
    """Read and return a list of file contents for each path provided."""
    resolved_paths = [
        os.path.abspath(os.path.normpath(f"{WORK_DIR}/{path}")) for path in file_paths
    ]
    contents = []
    for path in resolved_paths:
        with open(path, "r") as f:
            contents.append(f.read())
    return contents


def save_file(file_path: str, file_contents: str) -> str:
    """Create ``file_path`` relative to ``WORK_DIR`` with ``file_contents``."""
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    if os.path.exists(resolved_path):
        raise Exception(f"File already exists at {resolved_path}.")

    directory = os.path.dirname(resolved_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(resolved_path, "w") as f:
        f.write(file_contents)

    return f"File saved to {resolved_path}."


def save_multiple_files(file_paths: List[str], file_contents: List[str]) -> str:
    """Save each item in ``file_contents`` to the matching path."""
    resolved_paths = [
        os.path.abspath(os.path.normpath(f"{WORK_DIR}/{path}")) for path in file_paths
    ]

    for path in resolved_paths:
        if os.path.exists(path):
            raise Exception(f"File already exists at {path}.")

    for idx, path in enumerate(resolved_paths):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w") as f:
            f.write(file_contents[idx])

    return f"Files saved to {resolved_paths}."


# Agent dedicated to executing arbitrary code blocks












def execute_code_block(lang: str, code_block: str) -> str:
    """Execute a single code block and capture the execution logs.

    The first line may specify a file to save the code to using the format
    ``# filename: <name>``. In that case the code is written to the given
    path relative to :data:`WORK_DIR` before executing.
    """

    from agents import code_execution_agent  # imported here to avoid circular imports during tests

    lines = code_block.splitlines()
    save_path = None
    resolved = ""
    if lines and lines[0].lstrip().startswith("# filename:"):
        save_path = lines[0].split(":", 1)[1].strip()
        resolved = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{save_path}"))
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w") as f:
            f.write("\n".join(lines[1:]))
        # Remove the filename line when executing
        code_block = "\n".join(lines[1:])

    # Some versions of ``AssistantAgent`` do not expose ``_code_execution_config``
    # so we guard against its absence before attempting to modify it. This keeps
    # ``execute_code_block`` compatible with both old and new implementations of
    # the agent.
    if hasattr(code_execution_agent, "_code_execution_config"):
        code_execution_agent._code_execution_config.pop("last_n_messages", None)
    exitcode, logs = code_execution_agent.execute_code_blocks([(lang, code_block)])
    status = "execution succeeded" if exitcode == 0 else "execution failed"
    prefix = f"Saved code to {resolved}\n" if save_path else ""
    return f"{prefix}exitcode: {exitcode} ({status})\nCode output: {logs}"


