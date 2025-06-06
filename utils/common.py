import os
import json
import config


def _write_markdown_message(f, message):
    """Write a single message to a markdown file in readable format."""
    speaker = getattr(message, "speaker", None) or getattr(message, "role", None) or "unknown"
    content = getattr(message, "content", "")
    if isinstance(content, str):
        content = content.replace("\\n", "\n")
    f.write(f"## {speaker}\n{content}\n\n")


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
            json.dump(message.dict(), jf)
            jf.write("\n")
            jf.flush()
            _write_markdown_message(mf, message)
            mf.flush()
            yield message
