import os
import json
import config


def _get(message, key, default=None):
    """Helper to fetch an attribute from a dict or object."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _write_markdown_message(f, message):
    """Write a single message to a markdown file in readable format.

    Besides the speaker and content, if ``source`` or ``models_usage`` fields are
    present on the message, they are also written to the markdown log.
    """

    speaker = _get(message, "speaker") or _get(message, "role") or "unknown"
    content = _get(message, "content", "")
    if isinstance(content, str):
        content = content.replace("\\n", "\n")
    f.write(f"## {speaker}\n{content}\n")

    source = _get(message, "source")
    models_usage = _get(message, "models_usage")
    if source is not None or models_usage is not None:
        meta = {}
        if source is not None:
            meta["source"] = source
        if models_usage is not None:
            meta["models_usage"] = models_usage
        f.write("```json\n")
        f.write(json.dumps(meta, indent=2))
        f.write("\n```\n")
    f.write("\n")


def save_communication_log(messages):
    """Save a list of message dicts to conversation_log.jsonl.

    Parameters
    ----------
    messages : list
        List of dictionaries representing the conversation messages.

    Returns
    -------
    dict
        Dict with success status and path to the log file.
    """
    try:
        os.makedirs(config.GENERATED_FILES_DIR, exist_ok=True)
        jsonl_path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.jsonl")
        md_path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.md")
        with open(jsonl_path, "w", encoding="utf-8") as jf, open(md_path, "w", encoding="utf-8") as mf:
            for msg in messages:
                content = msg.get("content", "").replace("\\n", "\n")
                print(f"{msg.get('role', 'unknown')}: {content}\n")
                json.dump(msg, jf)
                jf.write("\n")
                _write_markdown_message(mf, msg)
        return {"success": True, "path": jsonl_path, "md_path": md_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def log_stream(stream):
    """Yield messages from an async stream and persist each one to disk."""
    os.makedirs(config.GENERATED_FILES_DIR, exist_ok=True)
    jsonl_path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.jsonl")
    md_path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.md")
    with open(jsonl_path, "w", encoding="utf-8") as jf, open(md_path, "w", encoding="utf-8") as mf:
        async for message in stream:
            if hasattr(message, "dict"):
                data = message.dict()
            elif isinstance(message, dict):
                data = message
            else:
                data = {"content": str(message)}
            json.dump(data, jf)
            jf.write("\n")
            jf.flush()
            _write_markdown_message(mf, message)
            mf.flush()
            yield message
