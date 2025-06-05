"""Utilities for retrieval augmented generation (RAG)."""

from __future__ import annotations

import json
import os
from typing import Dict, Any

import openai

import config
from prompts import ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT
from utils.search_tools import find_relevant_github_repo


def light_gpt4_wrapper_autogen(prompt: str) -> str:
    """Query OpenAI GPT-4 to obtain a JSON domain match result."""
    api_key = os.environ.get("gpt_api_key")
    if not api_key:
        raise RuntimeError("Missing gpt_api_key environment variable")

    response = openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]


def get_informed_answer(domain: str, question: str) -> str:
    """Answer a question using the domain description as context."""
    doc_path = os.path.join(
        config.DOMAIN_KNOWLEDGE_DOCS_DIR, domain, "domain_description.txt"
    )
    context = ""
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            context = f.read()

    if not context:
        return "No domain information available.".strip()

    api_key = os.environ.get("gpt_api_key")
    if not api_key:
        return context

    prompt = (
        "You are a knowledgeable assistant. Use the following domain "
        "description to answer the user's question as best as you can.\n\n" + context
    )
    response = openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
        temperature=0,
    )
    return response["choices"][0]["message"]["content"].strip()


def consult_archive_agent(question: str, threshold: float = 0.5) -> str:
    """Consult the local knowledge archive for a domain specific answer.

    Parameters
    ----------
    question:
        The question to ask the archive agent.
    threshold:
        Minimum rating for accepting a domain match.
    """
    domains: Dict[str, str] = {}
    docs_dir = config.DOMAIN_KNOWLEDGE_DOCS_DIR
    if os.path.isdir(docs_dir):
        for name in os.listdir(docs_dir):
            desc_path = os.path.join(docs_dir, name, "domain_description.txt")
            if os.path.exists(desc_path):
                with open(desc_path, "r", encoding="utf-8") as f:
                    domains[name] = f.read()

    prompt_parts = [ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT, "\n", question]
    for dname, ddesc in domains.items():
        prompt_parts.append(f"\nDOMAIN: {dname}\n{ddesc}\n")
    prompt = "".join(prompt_parts)

    try:
        response = light_gpt4_wrapper_autogen(prompt)
        result = json.loads(response or "{}")
        domain = result.get("domain", "")
        rating = float(result.get("match_rating", 0))
    except Exception:
        domain = ""
        rating = 0.0

    if rating < threshold:
        _, domain = find_relevant_github_repo(question)

    answer = get_informed_answer(domain=domain, question=question)
    return answer

