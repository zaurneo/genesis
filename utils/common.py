import os
import json
import config
import logging


def _get(message, key, default=None):
    """Helper to fetch an attribute from a dict or object."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _make_json_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'model_dump'):
        # Handle Pydantic models
        try:
            return obj.model_dump()
        except:
            return str(obj)
    elif hasattr(obj, 'dict'):
        # Handle other objects with dict() method
        try:
            return obj.dict()
        except:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dictionary
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()}
    else:
        # Convert other non-serializable objects to string
        return str(obj)


def _write_markdown_message(f, message):
    """Write a single message to a markdown file in readable format.

    Besides the speaker and content, if ``source`` or ``models_usage`` fields are
    present on the message, they are also written to the markdown log.
    """
    try:
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
            
            # Make meta JSON serializable
            serializable_meta = _make_json_serializable(meta)
            
            f.write("```json\n")
            try:
                f.write(json.dumps(serializable_meta, indent=2, ensure_ascii=False))
            except (TypeError, ValueError) as json_error:
                # Fallback if JSON serialization still fails
                f.write(json.dumps({
                    "serialization_error": str(json_error),
                    "meta_type": str(type(meta)),
                    "meta_str": str(meta)[:500]  # Truncate to prevent huge logs
                }, indent=2))
            f.write("\n```\n")
        f.write("\n")
        
    except Exception as e:
        # Ultimate fallback - write basic info
        logging.warning(f"Failed to write message to markdown: {e}")
        try:
            f.write(f"## Error\nFailed to write message: {str(e)}\n")
            f.write(f"Message type: {type(message)}\n\n")
        except:
            # If even this fails, we give up on this message
            pass


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
                try:
                    content = msg.get("content", "").replace("\\n", "\n")
                    print(f"{msg.get('role', 'unknown')}: {content}\n")
                    
                    # Make message JSON serializable before writing
                    serializable_msg = _make_json_serializable(msg)
                    json.dump(serializable_msg, jf)
                    jf.write("\n")
                    _write_markdown_message(mf, msg)
                except Exception as msg_error:
                    logging.warning(f"Failed to save message: {msg_error}")
                    # Write error info but continue processing
                    error_entry = {
                        "error": str(msg_error),
                        "original_type": str(type(msg)),
                        "content": str(msg)[:200]  # Truncated content
                    }
                    json.dump(error_entry, jf)
                    jf.write("\n")
                    
        return {"success": True, "path": jsonl_path, "md_path": md_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def log_stream(stream):
    """Yield messages from an async stream and persist each one to disk."""
    try:
        os.makedirs(config.GENERATED_FILES_DIR, exist_ok=True)
        jsonl_path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.jsonl")
        md_path = os.path.join(config.GENERATED_FILES_DIR, "conversation_log.md")
        
        with open(jsonl_path, "w", encoding="utf-8") as jf, open(md_path, "w", encoding="utf-8") as mf:
            async for message in stream:
                try:
                    # Convert message to dict safely
                    if hasattr(message, "dict"):
                        try:
                            data = message.dict()
                        except:
                            data = {"content": str(message), "type": str(type(message))}
                    elif isinstance(message, dict):
                        data = message
                    else:
                        data = {"content": str(message), "type": str(type(message))}
                    
                    # Make data JSON serializable
                    serializable_data = _make_json_serializable(data)
                    
                    # Write to JSONL
                    json.dump(serializable_data, jf)
                    jf.write("\n")
                    jf.flush()
                    
                    # Write to markdown
                    _write_markdown_message(mf, message)
                    mf.flush()
                    
                    # Yield the original message to continue the stream
                    yield message
                    
                except Exception as write_error:
                    # Log the error but continue processing
                    logging.error(f"Failed to write message to log: {write_error}")
                    
                    # Try to write a minimal error entry
                    try:
                        error_entry = {
                            "error": str(write_error),
                            "message_type": str(type(message)),
                            "timestamp": str(getattr(message, 'timestamp', 'unknown'))
                        }
                        json.dump(error_entry, jf)
                        jf.write("\n")
                        jf.flush()
                        
                        mf.write(f"<!-- Error writing message: {str(write_error)} -->\n\n")
                        mf.flush()
                    except:
                        # If even error logging fails, just continue
                        pass
                    
                    # Still yield the message to keep the stream alive
                    yield message
                    
    except Exception as file_error:
        logging.error(f"Failed to open log files: {file_error}")
        
        # If we can't write to files, just pass through the stream
        async for message in stream:
            yield message