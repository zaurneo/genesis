"""Utilities for searching external sources like GitHub."""

from __future__ import annotations

import os
from typing import Tuple
import json
from urllib import request, parse


def find_relevant_github_repo(query: str, max_results: int = 1) -> Tuple[str, str]:
    """Search GitHub repositories related to the given query.

    Parameters
    ----------
    query:
        Query string describing the desired repository.
    max_results:
        Number of repositories to retrieve and consider. Only the top result is returned.

    Returns
    -------
    Tuple[str, str]
        URL and full name of the best matching repository. Returns empty strings if none found.
    """
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}

    try:
        url = "https://api.github.com/search/repositories?" + parse.urlencode(params)
        req = request.Request(url, headers=headers)
        with request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        items = data.get("items") or []
        if not items:
            return "", ""
        top = items[0]
        return top.get("html_url", ""), top.get("full_name", "")
    except Exception:
        return "", ""
