import os
import json
import config


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
        path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for msg in messages:
                json.dump(msg, f)
                f.write("\n")
        return {"success": True, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def log_stream(stream):
    """Yield messages from an async stream and log them to disk."""
    messages = []
    async for message in stream:
        messages.append(message)
        yield message
    save_communication_log(messages)
