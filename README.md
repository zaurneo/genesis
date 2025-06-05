# Company Research Multi-Agent System

This is the modular structure.

## Owner Mediated Workflow

The `OwnerMediationGroupChat` ensures every agent interacts only with the
`Project_Owner`. Agents take turns in a round-robin sequence mediated by the
owner. Once all analytical tasks are complete the owner switches the chat to a
report phase. During this phase only the `Project_Owner` and the new
`Report_Insight_Generator` agent can speak. Attempting to start the report phase
before all tasks are marked completed will result in an error. The report agent
will refuse to generate the HTML summary unless the report phase has been
activated. Once active, the agent compiles `investor_report.html` using all
generated data and evaluations so investors receive a clear, structured summary.

## Environment Variables and Domain Knowledge

Several environment variables configure API access and where domain knowledge is stored:

- `gpt_api_key` – OpenAI API key.
- `claude_api_key` – Anthropic API key.
- `DOMAIN_KNOWLEDGE_DOCS_DIR` – directory containing domain descriptions (defaults to `knowledge/input`).
- `DOMAIN_KNOWLEDGE_STORAGE_DIR` – directory used to store indexed domain knowledge (defaults to `knowledge/output`).
- `GITHUB_TOKEN` – optional token for authenticated GitHub API requests used by `find_relevant_github_repo`.

Domain descriptions should be organised with the following layout:

```
knowledge/input/<domain>/domain_description.txt
```
